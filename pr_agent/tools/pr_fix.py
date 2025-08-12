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

        # Try child PR workflow first (hackathon MVP)
        child_pr_success = self.handle_fix_command()
        if child_pr_success:
            return  # Child PR workflow completed successfully

        # Fallback to existing workflow
        if get_settings().config.publish_output:
            self.git_provider.publish_comment("Generating fix patch...", is_temporary=True)

        # Collect PR context and files within scope limits
        files_ctx = self._collect_changed_files()
        if not files_ctx:
            self._publish_final("No eligible files to fix (check allowlist/blocklist and size limits).")
            return

        # If configured, try Aider CLI to apply fixes directly, then synthesize diff
        use_aider = bool(get_settings().get("pr_fix", {}).get("use_aider", False))
        if use_aider:
            ok, aider_patch, aider_err = self._run_aider(files_ctx)
            if not ok:
                self._publish_final(f"Aider failed: {aider_err}")
                return
            # Skip validation for aider - it's self-validating
            summary = self._build_summary(0.99, "aider-applied fix", files_ctx)
            comment = summary + "\n\n```diff\n" + aider_patch + "\n```\n\n"
            comment += "> Aider integration: Changes applied successfully and presented for review."
            self._publish_final(comment)
            return

        # Render prompts and call model
        system_prompt, user_prompt = self._render_prompts(files_ctx)

        # Build prompt
        # system_prompt, user_prompt = self._render_prompts(files_ctx)

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
        # Prefer ops-based edits if provided; fall back to raw patch
        edits = payload.get("edits")
        patch = str(payload.get("patch", "")).strip() if not edits else ""

        th = float(get_settings().get("pr_fix", {}).get("confidence_threshold", 0.85))
        if confidence < th:
            self._publish_final(f"Proposed fix confidence {confidence:.2f} is below threshold {th:.2f}. Aborting.")
            return

        synthesized_patch = None
        if edits:
            ok_build, synthesized_patch, build_err = self._build_patch_from_edits(edits, files_ctx)
            if not ok_build:
                self._publish_final(f"Failed to build patch from edits: {build_err}")
                return
        else:
            if not patch:
                self._publish_final("Model did not provide a patch.")
                return
            synthesized_patch = patch

        # Validate patch applies (dry-run)
        ok, err = self._dry_run_patch(synthesized_patch)
        if not ok:
            self._publish_final(f"Patch failed to apply in dry-run: {err}")
            return

        # For MVP, publish safe-mode comment with patch + summary
        summary = self._build_summary(confidence, rationale, files_ctx)
        comment = summary + "\n\n```diff\n" + synthesized_patch + "\n```\n\n"
        comment += "> Safe mode: posting patch as diff. Branch/PR delivery to follow."
        self._publish_final(comment)

    def _run_aider(self, files_ctx: list[dict]) -> tuple[bool, str | None, str | None]:
        """
        Invoke Aider CLI non-interactively to apply minimal fixes to the listed files.
        Returns (ok, synthesized_unified_diff, error)
        """
        try:
            import shutil
        except Exception:
            pass
        
        # First check for environment variable (GitHub Actions)
        aider_exe = os.environ.get("AIDER_EXECUTABLE")
        if not aider_exe:
            # Fallback to PATH search
            aider_exe = shutil.which("aider") if 'shutil' in globals() else None
        
        if not aider_exe or not os.path.exists(aider_exe):
            return False, None, "aider CLI not found. Ensure aider-chat is installed or AIDER_EXECUTABLE is set."

        # Build instruction from context
        title = getattr(self.git_provider.pr, 'title', '') if hasattr(self.git_provider, 'pr') else ''
        desc = self.git_provider.get_pr_description() if hasattr(self.git_provider, 'get_pr_description') else ''
        review_text = ""
        if bool(get_settings().get("pr_fix", {}).get("use_review_context", True)):
            try:
                prev = self.git_provider.get_previous_review(full=True, incremental=False)
                if prev and getattr(prev, "body", ""):
                    review_text = prev.body
            except Exception:
                pass

        files = [f.get("path") for f in files_ctx if isinstance(f, dict) and f.get("path")]
        files = files[: int(get_settings().get("pr_fix", {}).get("max_files", 10))]
        
        # Debug: Show what files are actually on disk vs what we think should exist
        get_logger().info(f"Working directory: {os.getcwd()}")
        
        # List all files in working directory
        import glob
        all_files = glob.glob("**/*", recursive=True)
        get_logger().info(f"All files on disk: {all_files[:20]}...")  # First 20 files
        
        # Check that files actually exist and are readable
        existing_files = []
        for file_path in files:
            abs_path = os.path.abspath(file_path)
            if os.path.exists(file_path):
                existing_files.append(file_path)
                get_logger().info(f"File exists: {file_path}")
            else:
                get_logger().warning(f"File does not exist: {file_path} (abs: {abs_path})")
        
        if not existing_files:
            return False, None, "no files exist to modify"
        
        files = existing_files
        get_logger().info(f"Files for aider: {files}")
        
        # Build specific instruction based on review context and common issues
        specific_instruction = self._build_specific_aider_instruction(title, desc, review_text, files_ctx)
        
        # Run aider with architect mode, no auto-commits, and specific instructions
        try:
            msg_arg = ["--message", specific_instruction]
            cmd = [aider_exe, "--yes", "--architect", "--no-auto-commits"] + msg_arg + files
            
            # Log what we're trying to do
            get_logger().info(f"Running aider command: {' '.join(cmd)}")
            get_logger().info(f"Aider instruction: {specific_instruction[:200]}...")
            get_logger().info(f"Files to fix: {files}")
            
            # Prepare environment for Aider (map OPENAI_KEY -> OPENAI_API_KEY, ANTHROPIC_KEY -> ANTHROPIC_API_KEY)
            env = os.environ.copy()
            if "OPENAI_API_KEY" not in env and env.get("OPENAI_KEY"):
                env["OPENAI_API_KEY"] = env["OPENAI_KEY"]
            if "ANTHROPIC_API_KEY" not in env and env.get("ANTHROPIC_KEY"):
                env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_KEY"]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            get_logger().info(f"Aider exit code: {proc.returncode}")
            
            # Log complete Aider output without truncation
            get_logger().info("=== FULL AIDER STDOUT START ===")
            get_logger().info(proc.stdout)
            get_logger().info("=== FULL AIDER STDOUT END ===")
            
            if proc.stderr:
                get_logger().info("=== FULL AIDER STDERR START ===")
                get_logger().info(proc.stderr)
                get_logger().info("=== FULL AIDER STDERR END ===")
            
            if proc.returncode != 0:
                return False, None, (proc.stderr or proc.stdout or "aider failed").strip()
        except Exception as e:
            get_logger().exception(f"Aider execution failed: {e}")
            return False, None, str(e)

        # Capture aider's changes - no validation needed since aider is self-validating
        try:
            # Aider successfully ran, so capture the changes it made
            diff_proc = subprocess.run(["git", "diff", "--no-prefix"], capture_output=True, text=True)
            if diff_proc.returncode != 0:
                return False, None, f"failed to capture diff: {diff_proc.stderr}"
            
            diff = diff_proc.stdout.strip()
            if not diff:
                return False, None, "aider produced no changes"
            
            get_logger().info(f"Aider diff output length: {len(diff)}")
            if diff:
                get_logger().info(f"Aider diff preview: {diff[:500]}...")
            
            # Ensure newline
            if not diff.endswith("\n"):
                diff += "\n"
                
            # Return diff for PR presentation - no validation needed
            return True, diff, None
                    
        except Exception as e:
            get_logger().exception(f"Aider diff capture failed: {e}")
            return False, None, str(e)

    def _build_specific_aider_instruction(self, title: str, desc: str, review_text: str, files_ctx: list[dict]) -> str:
        """
        Build specific, actionable instructions for Aider based on review context and common code issues.
        """
        # Extract specific actionable issues from review
        issues = self._extract_actionable_issues(review_text)
        
        if issues:
            instruction = "Fix the following specific issues identified in the code review:\n\n"
            for i, issue in enumerate(issues, 1):
                instruction += f"{i}. {issue}\n"
            instruction += "\n"
        else:
            # Fallback to common code issues when no specific review issues found
            instruction = "Fix the following types of issues if present in the code:\n\n"
            instruction += "1. Syntax errors, typos, and undefined variables\n"
            instruction += "2. Null pointer exceptions and missing null checks\n"
            instruction += "3. Type mismatches and incorrect casting\n"
            instruction += "4. Logic bugs that could cause runtime errors\n"
            instruction += "5. Security vulnerabilities (XSS, injection, etc.)\n"
            instruction += "6. Resource leaks (unclosed files, connections)\n"
            instruction += "7. Performance issues (inefficient loops, memory leaks)\n\n"
        
        # Add context
        instruction += f"PR Context:\n"
        instruction += f"Title: {title}\n"
        if desc.strip():
            instruction += f"Description: {desc}\n"
        
        # Add constraints
        instruction += "\nConstraints:\n"
        instruction += "- Make minimal, safe changes only\n"
        instruction += "- Do not refactor working code\n"
        instruction += "- Do not modify tests or CI/infrastructure files\n"
        instruction += "- Focus on correctness and safety fixes\n"
        instruction += "- Avoid speculative or stylistic changes\n"
        
        return instruction
    
    def _extract_actionable_issues(self, review_text: str) -> list[str]:
        """
        Extract specific, actionable issues from review text.
        """
        if not review_text:
            return []
        
        issues = []
        
        # Common patterns that indicate actionable issues
        issue_patterns = [
            r'(?i)(?:fix|correct|address|resolve)\s+(.+?)(?:\.|$)',
            r'(?i)(?:bug|error|issue|problem):\s*(.+?)(?:\.|$)',
            r'(?i)(?:should be|needs to be|must be)\s+(.+?)(?:\.|$)',
            r'(?i)(?:missing|lacking|without)\s+(.+?)(?:\.|$)',
            r'(?i)(?:incorrect|wrong|invalid)\s+(.+?)(?:\.|$)',
            r'(?i)(?:potential|possible)\s+(?:security|vulnerability|leak)\s*(.+?)(?:\.|$)',
        ]
        
        import re
        for pattern in issue_patterns:
            matches = re.findall(pattern, review_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                clean_issue = match.strip()
                if clean_issue and len(clean_issue) > 5 and len(clean_issue) < 200:
                    issues.append(clean_issue)
        
        # Limit to most relevant issues
        return issues[:5]

    def _build_patch_from_edits(self, edits: Any, context_files: list[dict]) -> tuple[bool, str | None, str | None]:
        """
        Build a unified diff (p0 paths) from structured edits.
        Supported minimal schema per edit:
          { "path": str, "type": "replace_snippet", "old": str, "new": str }
        Returns (ok, patch, error)
        """
        try:
            import difflib
        except Exception as e:
            return False, None, f"missing difflib: {e}"

        if not isinstance(edits, list):
            return False, None, "'edits' must be a list"

        updated_files: dict[str, str] = {}
        original_files: dict[str, str] = {}
        context_map: dict[str, str] = {f.get("path"): f.get("content", "") for f in (context_files or []) if isinstance(f, dict) and f.get("path")}

        def read_file(path: str) -> tuple[bool, str | None]:
            # Prefer disk if available; fallback to context_map (files from PR provider)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return True, f.read()
            except Exception:
                if path in context_map:
                    return True, context_map[path]
                # also try stripping leading './'
                alt = path[2:] if path.startswith("./") else None
                if alt and alt in context_map:
                    return True, context_map[alt]
                return False, f"[Errno 2] No such file or directory: '{path}'"

        # Apply edits in-memory
        for i, e in enumerate(edits):
            if not isinstance(e, dict):
                return False, None, f"edit #{i} is not an object"
            path = e.get("path")
            etype = e.get("type")
            if not path or not etype:
                return False, None, f"edit #{i} missing 'path' or 'type'"
            okr, content_or_err = read_file(path)
            if not okr:
                return False, None, f"cannot read {path}: {content_or_err}"
            current = content_or_err  # type: ignore[assignment]
            original_files.setdefault(path, current)

            if etype == "replace_snippet":
                old = e.get("old")
                new = e.get("new")
                if old is None or new is None:
                    return False, None, f"edit #{i} replace_snippet requires 'old' and 'new'"
                if old not in current:
                    return False, None, f"edit #{i} old snippet not found in {path}"
                updated = current.replace(old, new, 1)
                updated_files[path] = updated
            elif etype == "replace_line_range":
                start = e.get("start_line")
                end = e.get("end_line")
                new = e.get("new")
                if not isinstance(start, int) or not isinstance(end, int) or new is None:
                    return False, None, f"edit #{i} replace_line_range requires int start_line/end_line and 'new'"
                lines = current.splitlines()
                if start < 1 or end < start or end > len(lines) + 1:
                    return False, None, f"edit #{i} invalid line range for {path}"
                before = lines[: start - 1]
                after = lines[end - 1 :]
                new_lines = new.splitlines()
                updated = "\n".join(before + new_lines + after)
                updated_files[path] = updated
            else:
                return False, None, f"edit #{i} unsupported type: {etype}"

        # Build unified diff across edited files
        patches: list[str] = []
        for path, new_content in updated_files.items():
            old_content = original_files.get(path, "")
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            # Use path as-is (p0 paths)
            ud = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=path, tofile=path,
                lineterm=""
            )
            diff_text = "\n".join(list(ud))
            if diff_text:
                patches.append(diff_text)

        full_patch = "\n".join(patches)
        if not full_patch.strip():
            return False, None, "no changes produced by edits"
        # Ensure trailing newline for git apply
        if not full_patch.endswith("\n"):
            full_patch += "\n"
        return True, full_patch, None

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
        # If association requirement is disabled, allow any commenter.
        try:
            require = bool(get_settings().get("pr_fix", {}).get("require_association", False))
        except Exception:
            require = False
        if not require:
            return True
        # Otherwise enforce allowed associations (MEMBER/OWNER/COLLABORATOR)
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
        # Optionally include prior review as guidance
        use_review = bool(get_settings().get("pr_fix", {}).get("use_review_context", True))
        review_text = ""
        if use_review:
            try:
                prev = self.git_provider.get_previous_review(full=True, incremental=False)
                review_text = prev.body if prev else ""
            except Exception:
                review_text = ""
        vars = {
            "title": title,
            "description": desc,
            "files": files_ctx,
            "review": review_text,
        }
        system = get_settings().get("pr_fix_prompt", {}).get("system", "You are a code-fixing assistant.")
        user = get_settings().get("pr_fix_prompt", {}).get("user", "")
        # Very light templating
        user_rendered = user.replace("{{title}}", title or "").replace("{{description}}", desc or "")
        if use_review:
            user_rendered = user_rendered.replace("{{review}}", review_text or "")
        else:
            user_rendered = user_rendered.replace("{{review}}", "")
        files_blob = "\n\n".join([f"# {f['path']}\n{f['content']}" for f in files_ctx])
        user_rendered = user_rendered.replace("{{files}}", files_blob)
        return system, user_rendered

    async def _chat(self, model: str, system_prompt: str, user_prompt: str):
        resp, finish = await self.ai_handler.chat_completion(
            model=model, system=system_prompt, user=user_prompt, temperature=get_settings().config.temperature
        )
        return resp, finish

    def _dry_run_patch(self, patch: str) -> tuple[bool, str | None]:
        # Sanitize patch and validate with git apply --check. Try -p0, then fallback to -p1
        try:
            # 1) Sanitize: strip code fences/backticks and extra prose
            p = patch.strip()
            # remove ```diff ... ``` or ``` ... ``` fences if present
            if p.startswith("```"):
                # drop leading fence line
                first_nl = p.find("\n")
                if first_nl != -1:
                    p = p[first_nl + 1 :]
            if p.endswith("```"):
                p = p[: -3].rstrip()
            # strip possible JSON escaping artifacts
            p = p.replace("\r\n", "\n").replace("\r", "\n")
            # remove byte-order mark
            if p and p[0] == "\ufeff":
                p = p.lstrip("\ufeff")
            # ensure trailing newline (git apply can be picky)
            if not p.endswith("\n"):
                p += "\n"

            # Heuristic: detect a/ b/ path prefixes or diff --git headers
            has_ab_prefix = False
            for line in p.splitlines()[:10]:
                if line.startswith("diff --git") or line.startswith("--- a/") or line.startswith("+++ b/"):
                    has_ab_prefix = True
                    break

            def attempt(patch_text: str, strip_level: str) -> tuple[bool, str | None]:
                with tempfile.TemporaryDirectory() as td:
                    patch_path = os.path.join(td, "fix.patch")
                    with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(patch_text)
                    cmd = ["git", "apply", "--check", strip_level, patch_path]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        return True, None
                    return False, (proc.stderr or proc.stdout or "patch apply failed").strip()

            # Try -p0 first
            ok, err = attempt(p, "-p0")
            if ok:
                return True, None
            # Fallback to -p1 if patch seems to use a/ b/ prefixes
            if has_ab_prefix:
                ok2, err2 = attempt(p, "-p1")
                if ok2:
                    return True, None
                return False, f"-p0 failed: {err}; -p1 failed: {err2}"
            return False, err
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

    # ===== CHILD PR WORKFLOW METHODS (Hackathon MVP) =====

    def handle_fix_command(self) -> bool:
        """
        Handle /fix command with child PR workflow.
        Returns True if child PR was created successfully, False otherwise.
        """
        # Check if child PR workflow is enabled
        child_pr_settings = get_settings().get("pr_fix", {}).get("child_pr", {})
        if not child_pr_settings.get("enabled", False):
            get_logger().info("Child PR workflow disabled, falling back to standard flow")
            return False

        try:
            # Get PR details
            pr_number = self.git_provider.pr.number
            parent_branch = self.git_provider.pr.base.ref
            
            # Create unique fix branch
            import time
            timestamp = int(time.time())
            fix_branch = f"fix-{pr_number}-{timestamp}"
            
            get_logger().info(f"Creating fix branch: {fix_branch}")
            
            # Create and checkout fix branch
            subprocess.run(["git", "checkout", "-b", fix_branch], check=True)
            
            # Collect files and run Aider fixes
            files_ctx = self._collect_changed_files()
            if not files_ctx:
                self._publish_final("No eligible files to fix (check allowlist/blocklist and size limits).")
                return False
            
            # Run Aider with architect mode
            ok, aider_patch, aider_err = self._run_aider(files_ctx)
            if not ok:
                get_logger().error(f"Aider failed: {aider_err}")
                # Cleanup and fallback
                subprocess.run(["git", "checkout", parent_branch], check=False)
                subprocess.run(["git", "branch", "-D", fix_branch], check=False)
                return False
            
            # Commit changes
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "🤖 Initial AI fixes"], check=True)
            subprocess.run(["git", "push", "origin", fix_branch], check=True)
            
            # Create child PR
            child_pr_number = self._create_child_pr(fix_branch, parent_branch, pr_number)
            if not child_pr_number:
                get_logger().error("Failed to create child PR")
                return False
            
            # Comment on parent PR
            self._notify_parent_pr(pr_number, child_pr_number)
            
            get_logger().info(f"Successfully created child PR #{child_pr_number}")
            return True
            
        except Exception as e:
            get_logger().exception(f"Child PR creation failed: {e}")
            # Try to cleanup
            try:
                subprocess.run(["git", "checkout", parent_branch], check=False)
                subprocess.run(["git", "branch", "-D", fix_branch], check=False)
            except:
                pass
            return False

    def _create_child_pr(self, fix_branch: str, parent_branch: str, parent_pr_number: int) -> int:
        """Create child PR and return its number, or 0 if failed."""
        try:
            child_pr_settings = get_settings().get("pr_fix", {}).get("child_pr", {})
            draft_mode = child_pr_settings.get("draft_mode", True)
            
            title = f"🔧 AI Fixes for PR #{parent_pr_number}"
            body = self._create_child_pr_template(parent_pr_number)
            
            # Create PR via git provider
            child_pr = self.git_provider.github_client.pulls.create(
                title=title,
                head=fix_branch,
                base=parent_branch,
                body=body,
                draft=draft_mode
            )
            
            return child_pr.number
            
        except Exception as e:
            get_logger().exception(f"Failed to create child PR: {e}")
            return 0

    def _create_child_pr_template(self, parent_pr_number: int) -> str:
        """Generate child PR template."""
        return f"""## 🤖 AI-Generated Fixes for PR #{parent_pr_number}

### How This Works
This PR contains AI-generated improvements to your code. Here's how to use it:

1. **📝 Review the changes** line by line
2. **💬 Leave comments** if you want modifications  
3. **✅ Approve when satisfied** - I'll merge it back automatically

### What I Fixed
I analyzed your PR and applied fixes using Aider with architect mode to address:
- Code issues identified in the review
- Common patterns and best practices
- Syntax errors and type issues

### Conversation Log
This PR will be updated as I address your feedback. Each commit represents a round of improvements.

---
🤖 *Generated by FixItFast AI - Let's make your code better together!*
"""

    def _notify_parent_pr(self, parent_pr_number: int, child_pr_number: int):
        """Add notification comment to parent PR."""
        comment = f"""🤖 I've analyzed your PR and created fixes in **PR #{child_pr_number}**

Please review the changes and let me know if you'd like any adjustments!

[View the fixes →](../../pull/{child_pr_number})
"""
        self.git_provider.publish_comment(comment)

    def handle_child_pr_comment(self, child_pr_number: int, comment_body: str) -> bool:
        """
        Handle comments on child PRs for iterative improvements.
        Returns True if handled successfully.
        """
        try:
            # Check for approval signals
            approval_words = ['lgtm', 'looks good', 'approve', 'approved', 'merge']
            if any(word in comment_body.lower() for word in approval_words):
                return self._merge_child_pr_to_parent(child_pr_number)
            
            # Otherwise, treat as feedback for improvements
            return self._apply_feedback_to_child_pr(child_pr_number, comment_body)
            
        except Exception as e:
            get_logger().exception(f"Failed to handle child PR comment: {e}")
            return False

    def _apply_feedback_to_child_pr(self, child_pr_number: int, feedback: str) -> bool:
        """Apply user feedback to child PR by adding commits."""
        try:
            # Get child PR details
            child_pr = self.git_provider.github_client.pulls.get(child_pr_number)
            fix_branch = child_pr.head.ref
            
            # Checkout the fix branch
            subprocess.run(["git", "fetch", "origin"], check=True)
            subprocess.run(["git", "checkout", fix_branch], check=True)
            
            # Get files context for Aider
            files_ctx = self._collect_changed_files()
            if not files_ctx:
                return False
            
            # Run Aider with specific feedback
            ok, aider_patch, aider_err = self._run_aider_with_feedback(files_ctx, feedback)
            if not ok:
                get_logger().error(f"Aider feedback processing failed: {aider_err}")
                return False
            
            # Commit improvements to same branch
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run([
                "git", "commit", "-m", 
                f"🤖 Address feedback: {feedback[:50]}..."
            ], check=True)
            subprocess.run(["git", "push", "origin", fix_branch], check=True)
            
            # Add acknowledgment comment
            self.git_provider.github_client.issues.create_comment(
                issue_number=child_pr_number,
                body=f"✅ I've updated the code based on your feedback:\n\n"
                     f"> {feedback}\n\n"
                     f"Please review the latest changes!"
            )
            
            return True
            
        except Exception as e:
            get_logger().exception(f"Failed to apply feedback: {e}")
            # Add error comment
            try:
                self.git_provider.github_client.issues.create_comment(
                    issue_number=child_pr_number,
                    body=f"❌ Sorry, I had trouble processing your feedback: {str(e)}"
                )
            except:
                pass
            return False

    def _run_aider_with_feedback(self, files_ctx: list[dict], feedback: str) -> tuple[bool, str | None, str | None]:
        """Run Aider with specific user feedback."""
        try:
            import shutil
        except Exception:
            pass
        
        # Get Aider executable
        aider_exe = os.environ.get("AIDER_EXECUTABLE")
        if not aider_exe:
            aider_exe = shutil.which("aider") if 'shutil' in globals() else None
        
        if not aider_exe or not os.path.exists(aider_exe):
            return False, None, "aider CLI not found"

        files = [f.get("path") for f in files_ctx if isinstance(f, dict) and f.get("path")]
        existing_files = [f for f in files if os.path.exists(f)]
        
        if not existing_files:
            return False, None, "no files exist to modify"
        
        # Build instruction from feedback
        instruction = f"Address the following user feedback:\n\n{feedback}\n\n"
        instruction += "Make the requested changes while maintaining code quality and existing functionality."
        
        try:
            cmd = [aider_exe, "--yes", "--architect", "--no-auto-commits", "--message", instruction] + existing_files
            
            # Prepare environment
            env = os.environ.copy()
            if "OPENAI_API_KEY" not in env and env.get("OPENAI_KEY"):
                env["OPENAI_API_KEY"] = env["OPENAI_KEY"]
            if "ANTHROPIC_API_KEY" not in env and env.get("ANTHROPIC_KEY"):
                env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_KEY"]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if proc.returncode != 0:
                return False, None, (proc.stderr or proc.stdout or "aider failed").strip()
            
            # Capture changes
            diff_proc = subprocess.run(["git", "diff", "--no-prefix"], capture_output=True, text=True)
            if diff_proc.returncode != 0:
                return False, None, f"failed to capture diff: {diff_proc.stderr}"
            
            diff = diff_proc.stdout.strip()
            if not diff:
                return False, None, "aider produced no changes"
            
            return True, diff, None
            
        except Exception as e:
            return False, None, str(e)

    def _merge_child_pr_to_parent(self, child_pr_number: int) -> bool:
        """Merge child PR back to parent PR."""
        try:
            # Merge child PR using GitHub API
            self.git_provider.github_client.pulls.merge(
                pull_number=child_pr_number,
                merge_method="squash",
                commit_title="🤖 Apply AI fixes",
                commit_message="Applied AI-generated fixes after review"
            )
            
            # Clean up fix branch
            child_pr = self.git_provider.github_client.pulls.get(child_pr_number)
            self.git_provider.github_client.git.delete_ref(f"heads/{child_pr.head.ref}")
            
            get_logger().info(f"Successfully merged child PR #{child_pr_number}")
            return True
            
        except Exception as e:
            get_logger().exception(f"Failed to merge child PR: {e}")
            # Add manual merge instructions comment
            try:
                self.git_provider.github_client.issues.create_comment(
                    issue_number=child_pr_number,
                    body=f"❌ I couldn't merge automatically: {str(e)}\n\n"
                         f"Please merge this PR manually when ready."
                )
            except:
                pass
            return False
