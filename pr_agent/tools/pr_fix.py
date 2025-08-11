import json
import os
import re
import tempfile
import subprocess
from functools import partial
from typing import Dict, Any, List

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.pr_processing import retry_with_fallback_models
from pr_agent.algo.utils import ModelType
from pr_agent.config_loader import get_settings
from pr_agent.git_providers import get_git_provider
from pr_agent.log import get_logger


ALLOWED_ASSOCIATIONS = {"MEMBER", "OWNER", "COLLABORATOR"}


class PRFix:
    """
    Implements the /fix command for GitHub as per hack.md.

    MVP delivery strategy: Safe mode (publish patch as a comment) while enforcing
    gates and confidence threshold. Branch + child PR flow can be added next.
    """

    def __init__(self, pr_url: str, cli_mode: bool = False, args=None, ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        self.git_provider = get_git_provider()(pr_url)
        self.ai_handler = ai_handler()
        self.cli_mode = cli_mode
        self.args = args or []
        self.prediction: Dict[str, Any] | None = None

    async def run(self):
        # GitHub-only for v1
        if get_settings().config.git_provider != "github":
            get_logger().error("/fix currently supports GitHub only")
            if get_settings().config.publish_output:
                self.git_provider.publish_comment("/fix currently supports GitHub only")
            return

        # Approval gate: label + author association
        if not self._has_required_label():
            msg = "`/fix` requires the 'autofix-approved' label. Add the label and re-run."
            get_logger().info(msg)
            if get_settings().config.publish_output:
                self.git_provider.publish_comment(msg)
            return
        if not self._is_authorized_commenter():
            msg = "`/fix` can be executed only by MEMBER/OWNER/COLLABORATOR."
            get_logger().info(msg)
            if get_settings().config.publish_output:
                self.git_provider.publish_comment(msg)
            return

        if get_settings().config.publish_output:
            self.git_provider.publish_comment("Generating fix patch...", is_temporary=True)

        # Collect PR context and files within scope limits
        files_ctx = self._collect_changed_files()
        if not files_ctx:
            self._publish_final("No eligible files to fix (check allowlist/blocklist and size limits).")
            return

        # Build prompt
        system_prompt, user_prompt = self._render_prompts(files_ctx)

        # Call LLM with retry and parse JSON
        try:
            response, _ = await retry_with_fallback_models(
                lambda model: self._chat(model, system_prompt, user_prompt), model_type=ModelType.REGULAR
            )
        except Exception as e:
            get_logger().exception(f"LLM call failed: {e}")
            self._publish_final("LLM call failed. See logs.")
            return

        if not response:
            self._publish_final("Empty response from model.")
            return

        try:
            payload = json.loads(response)
        except Exception:
            # Try to extract JSON between code fences
            m = re.search(r"\{[\s\S]*\}", response)
            if not m:
                self._publish_final("Model response was not valid JSON.")
                return
            try:
                payload = json.loads(m.group(0))
            except Exception:
                self._publish_final("Model response JSON parse failed.")
                return

        confidence = float(payload.get("confidence", 0))
        rationale = str(payload.get("rationale", ""))
        patch = str(payload.get("patch", "")).strip()

        th = float(get_settings().get("pr_fix", {}).get("confidence_threshold", 0.85))
        if confidence < th:
            self._publish_final(f"Proposed fix confidence {confidence:.2f} is below threshold {th:.2f}. Aborting.")
            return

        if not patch:
            self._publish_final("Model did not provide a patch.")
            return

        # Validate patch applies (dry-run)
        ok, err = self._dry_run_patch(patch)
        if not ok:
            self._publish_final(f"Patch failed to apply in dry-run: {err}")
            return

        # For MVP, publish safe-mode comment with patch + summary
        summary = self._build_summary(confidence, rationale, files_ctx)
        comment = summary + "\n\n```diff\n" + patch + "\n```\n\n"
        comment += "> Safe mode: posting patch as diff. Branch/PR delivery to follow."
        self._publish_final(comment)

    def _publish_final(self, msg: str):
        try:
            self.git_provider.remove_initial_comment()
        except Exception:
            pass
        if get_settings().config.publish_output:
            self.git_provider.publish_comment(msg)

    def _has_required_label(self) -> bool:
        try:
            labels = self.git_provider.get_pr_labels(update=True)
            required = get_settings().get("pr_fix", {}).get("require_label", "autofix-approved")
            return required in labels
        except Exception:
            return False

    def _is_authorized_commenter(self) -> bool:
        try:
            assoc = getattr(self.git_provider.pr, "author_association", "NONE")
            return assoc in ALLOWED_ASSOCIATIONS
        except Exception:
            return False

    def _collect_changed_files(self) -> List[Dict[str, Any]]:
        allow_exts = set(get_settings().get("pr_fix", {}).get("allow_extensions",
                          [".py", ".js", ".ts", ".tsx", ".java", ".go", ".rb", ".sh", ".yaml", ".yml", ".json"]))
        block_paths = get_settings().get("pr_fix", {}).get("block_paths", [
            ".github/workflows/**", "infra/**", "terraform/**", "k8s/**", "secrets/**"
        ])
        max_files = int(get_settings().get("pr_fix", {}).get("max_files", 5))
        max_chars = int(get_settings().get("pr_fix", {}).get("max_file_chars", 12000))

        def is_blocked(path: str) -> bool:
            # Simple prefix-based check for MVP; can replace with glob later
            for pat in block_paths:
                if pat.endswith("/**"):
                    if path.startswith(pat[:-3]):
                        return True
                elif path.startswith(pat):
                    return True
            return False

        files = []
        try:
            for f in self.git_provider.get_diff_files():
                if is_blocked(f.filename):
                    continue
                if not any(f.filename.endswith(ext) for ext in allow_exts):
                    continue
                new_str = f.new_file_content_str if hasattr(f, 'new_file_content_str') else getattr(f, 'new_file_content', "")
                if not new_str:
                    new_str = getattr(f, 'new_file_content', "") or getattr(f, 'new_file_content_str', "")
                # Fallback: use patch context if full content isn't loaded
                content = new_str or f.patch or ""
                if not content:
                    continue
                content = content[:max_chars]
                files.append({"path": f.filename, "content": content})
                if len(files) >= max_files:
                    break
        except Exception as e:
            get_logger().exception(f"Failed to collect diff files: {e}")
            return []

        return files

    def _render_prompts(self, files_ctx: List[Dict[str, Any]]):
        title = self.git_provider.pr.title if getattr(self.git_provider, 'pr', None) else ""
        desc = self.git_provider.get_pr_description() if hasattr(self.git_provider, 'get_pr_description') else ""
        vars = {
            "title": title,
            "description": desc,
            "files": files_ctx,
        }
        system = get_settings().get("pr_fix_prompt", {}).get("system", "You are a code-fixing assistant.")
        user = get_settings().get("pr_fix_prompt", {}).get("user", "")
        # Very light templating
        user_rendered = user.replace("{{title}}", title or "").replace("{{description}}", desc or "")
        files_blob = "\n\n".join([f"# {f['path']}\n{f['content']}" for f in files_ctx])
        user_rendered = user_rendered.replace("{{files}}", files_blob)
        return system, user_rendered

    async def _chat(self, model: str, system_prompt: str, user_prompt: str):
        resp, finish = await self.ai_handler.chat_completion(
            model=model, system=system_prompt, user=user_prompt, temperature=get_settings().config.temperature
        )
        return resp, finish

    def _dry_run_patch(self, patch: str) -> tuple[bool, str | None]:
        # Write patch to temp file and run git apply --check -p0
        try:
            with tempfile.TemporaryDirectory() as td:
                patch_path = os.path.join(td, "fix.patch")
                with open(patch_path, "w", encoding="utf-8") as f:
                    f.write(patch)
                cmd = ["git", "apply", "--check", "-p0", patch_path]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    return False, proc.stderr.strip() or proc.stdout.strip()
            return True, None
        except Exception as e:
            return False, str(e)

    def _build_summary(self, confidence: float, rationale: str, files_ctx: List[Dict[str, Any]]) -> str:
        files_list = "\n".join([f"- {f['path']}" for f in files_ctx])
        return (
            f"FixItFast proposal\n\n"
            f"- Confidence: {confidence:.2f}\n"
            f"- Rationale: {rationale}\n"
            f"- Files considered (capped):\n{files_list}\n"
        )
