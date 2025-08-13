import os
import re
import time
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


class PRIssueFix:
    """
    Implements the /fix_issue command for GitHub issues.
    
    Hackathon MVP: Analyze GitHub issues and create child PRs with AI-generated fixes.
    """

    def __init__(self, issue_url: str, cli_mode: bool = False, args=None, ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        self.git_provider = get_git_provider()(issue_url)
        self.ai_handler = ai_handler()
        self.cli_mode = cli_mode
        self.args = args or []
        self.issue_url = issue_url
        self.issue = None

    async def run(self):
        """Main entry point for issue fix workflow."""
        try:
            # GitHub-only for hackathon MVP
            if get_settings().config.git_provider != "github":
                get_logger().error("/fix_issue currently supports GitHub only")
                return

            # Get the issue
            self.issue = self._get_issue_from_url()
            if not self.issue:
                get_logger().error("Could not retrieve issue from URL")
                return

            get_logger().info(f"Processing issue #{self.issue.number}: {self.issue.title}")

            # For hackathon: Skip complex auth checks, just try to fix
            # Analyze issue and find relevant files
            analysis = await self._analyze_issue()
            if not analysis:
                get_logger().error("Failed to analyze issue")
                return

            files = await self._find_relevant_files(analysis, analysis.get('keywords', []))
            if not files:
                get_logger().error("No relevant files found for issue")
                # Inform user about the issue
                try:
                    self.issue.create_comment(
                        f" **Unable to locate relevant files for this issue**\n\n"
                        f"**Issue Analysis:**\n"
                        f"- **Type:** {analysis.get('issue_type', 'Unknown')}\n"
                        f"- **Confidence:** {analysis.get('confidence', 0):.0%}\n"
                        f"- **Summary:** {analysis.get('summary', 'No summary available')}\n\n"
                        f"**Problem:** I couldn't find the files mentioned in the issue description in this repository.\n\n"
                        f"**Possible reasons:**\n"
                        f"- The file paths mentioned don't exist in this repository\n"
                        f"- The files might be in a different repository\n"
                        f"- The paths might have changed since the issue was created\n"
                        f"- The repository structure might be different than expected\n\n"
                        f"**Please help by:**\n"
                        f"- Verifying the correct file paths in this repository\n"
                        f"- Updating the issue description with accurate paths\n"
                        f"- Checking if this issue belongs to a different repository\n"
                        f"- Providing more specific file names or directories to investigate\n\n"
                        f"Once you provide the correct file paths, I'll be happy to analyze and fix the performance issue! "
                    )
                except Exception as comment_err:
                    get_logger().error(f"Failed to add no-files-found comment: {comment_err}")
                return

            # Create child PR with fix
            success = await self._create_issue_fix_pr(analysis, files)
            if success:
                get_logger().info("Successfully created issue fix PR")
            else:
                get_logger().error("Failed to create issue fix PR")

        except Exception as e:
            get_logger().exception(f"Failed to process issue fix: {e}")

    def _get_issue_from_url(self):
        """Extract issue from URL using GitHub provider."""
        try:
            # The GitHub provider should already be initialized with the issue URL
            if hasattr(self.git_provider, 'issue_main') and self.git_provider.issue_main:
                return self.git_provider.issue_main
            
            # Fallback: parse URL manually using provider's method
            repo_path, issue_number = self.git_provider._parse_issue_url(self.issue_url)
            if repo_path and issue_number:
                repo_obj = self.git_provider.github_client.get_repo(repo_path)
                return repo_obj.get_issue(issue_number)
            
            return None
        except Exception as e:
            get_logger().exception(f"Failed to get issue from URL: {e}")
            return None


    async def _analyze_issue(self) -> Dict[str, Any]:
        """Analyze the issue to determine what to fix."""
        try:
            # Simple analysis prompt for hackathon
            analysis_prompt = f"""
Analyze this GitHub issue and determine if it can be fixed with code changes:

ISSUE TITLE: {self.issue.title}

ISSUE DESCRIPTION:
{self.issue.body or 'No description provided'}

ISSUE LABELS: {[label.name for label in self.issue.labels]}

Respond with JSON containing:
{{
    "fixable": true/false,
    "confidence": 0.0-1.0,
    "issue_type": "bug_fix|feature_add|refactor|documentation",
    "summary": "brief description of what needs to be fixed",
    "keywords": ["list", "of", "relevant", "keywords"]
}}
"""

            try:
                response, _ = await retry_with_fallback_models(
                    lambda model: self._chat_analysis(model, analysis_prompt), 
                    model_type=ModelType.REGULAR
                )
                
                # Try to parse JSON response
                import json
                return json.loads(response)
            except Exception as e:
                get_logger().warning(f"LLM analysis failed, using fallback: {e}")
                
                # Fallback: simple heuristic analysis
                return self._fallback_analysis()

        except Exception as e:
            get_logger().exception(f"Issue analysis failed: {e}")
            return None

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        keywords = []
        issue_text = (self.issue.title + " " + (self.issue.body or "")).lower()
        
        # Extract potential file references
        file_pattern = r'\b\w+\.\w+\b'
        potential_files = re.findall(file_pattern, issue_text)
        
        # Extract common programming keywords
        common_keywords = ['error', 'bug', 'fix', 'function', 'method', 'class', 'api', 'auth', 'login', 'database']
        for keyword in common_keywords:
            if keyword in issue_text:
                keywords.append(keyword)
        
        return {
            "fixable": True,  # Optimistic for hackathon
            "confidence": 0.7,
            "issue_type": "bug_fix",
            "summary": f"Fix issue: {self.issue.title}",
            "keywords": keywords + potential_files
        }

    async def _find_relevant_files(self, analysis: dict, keywords: list[str]) -> list[str]:
        """Find relevant files for the issue using LLM intelligence."""
        try:
            files = []
            
            # Strategy 1: Look for explicit file mentions in issue (only exact matches)
            issue_text = self.issue.title + " " + (self.issue.body or "")
            file_pattern = r'\b[\w/.-]+\.[a-zA-Z]{1,4}\b'
            potential_files = re.findall(file_pattern, issue_text)
            
            for file_path in potential_files:
                if os.path.exists(file_path) and self._is_valid_code_file(file_path):
                    files.append(file_path)
                    get_logger().info(f"Found explicitly mentioned file: {file_path}")
            
            # Strategy 2: LLM-powered intelligent file discovery
            llm_files = await self._discover_relevant_files_with_llm(analysis)
            if llm_files:
                files.extend(llm_files)
                get_logger().info(f"LLM discovered {len(llm_files)} relevant files")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for f in files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            
            # Limit to max files for safety
            max_files = 5  # Hackathon limit
            unique_files = unique_files[:max_files]
            
            get_logger().info(f"Final file selection: {unique_files}")
            return unique_files

        except Exception as e:
            get_logger().exception(f"File discovery failed: {e}")
            return []

    def _is_valid_code_file(self, file_path: str) -> bool:
        """Check if file path looks like a valid code file."""
        # Use the same comprehensive list as in file discovery
        code_extensions = [
            '.py', '.js', '.ts', '.tsx', '.jsx',           # Web/Python
            '.java', '.scala', '.sbt', '.gradle',         # JVM languages  
            '.go', '.rs', '.cpp', '.c', '.h',             # Systems languages
            '.rb', '.php', '.swift', '.kt',               # Other languages
            '.sql', '.yaml', '.yml', '.json', '.toml'     # Config files
        ]
        return any(file_path.endswith(ext) for ext in code_extensions)



    async def _discover_relevant_files_with_llm(self, analysis: dict) -> list[str]:
        """Use LLM to intelligently discover relevant files for the issue."""
        try:
            from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
            
            ai_handler = LiteLLMAIHandler()
            
            # Build comprehensive repository context
            repo_structure = self._get_repository_structure()
            issue_context = f"Title: {self.issue.title}\nBody: {self.issue.body or 'No description'}"
            
            prompt = f"""You are a code analysis expert. Given this GitHub issue and repository structure, identify the most relevant files that need to be examined or modified to fix this issue.

**Issue:**
{issue_context}

**Repository Structure:**
{repo_structure}

**Analysis Context:**
- Issue Type: {analysis.get('issue_type', 'Unknown')}
- Summary: {analysis.get('summary', 'No summary')}
- Keywords: {', '.join(analysis.get('keywords', []))}

**Instructions:**
1. Analyze the issue description and identify what functionality/components are affected
2. Look at the repository structure to find files that likely contain the relevant code
3. Focus on files that would need to be modified to fix this specific issue
4. Prioritize source code files over configuration files
5. Return ONLY the file paths, one per line, no explanations
6. Maximum 5 files
7. Only return files that actually exist in the repository structure shown above

**Response Format:**
path/to/file1.py
path/to/file2.js
path/to/file3.java"""

            response = await ai_handler.chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                get_logger().warning("LLM file discovery: No response from AI")
                return []
            
            content = response.choices[0].message.content.strip()
            if not content:
                get_logger().warning("LLM file discovery: Empty response from AI")
                return []
            
            # Parse file paths from response
            file_paths = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('*'):
                    # Clean up any markdown formatting
                    if line.startswith('- '):
                        line = line[2:]
                    if line.startswith('`') and line.endswith('`'):
                        line = line[1:-1]
                    
                    # Validate file exists and is a code file
                    if os.path.exists(line) and self._is_valid_code_file(line):
                        file_paths.append(line)
                        get_logger().info(f"LLM suggested relevant file: {line}")
                    else:
                        get_logger().debug(f"LLM suggested non-existent file: {line}")
            
            return file_paths[:5]  # Limit to 5 files
            
        except Exception as e:
            get_logger().exception(f"LLM file discovery failed: {e}")
            return []

    def _get_repository_structure(self, max_depth: int = 3) -> str:
        """Get a concise view of repository structure for LLM analysis."""
        try:
            import os
            structure_lines = []
            
            def add_directory(path, depth=0, max_items=10):
                if depth > max_depth:
                    return
                
                indent = "  " * depth
                items = []
                
                try:
                    for item in sorted(os.listdir(path)):
                        if item.startswith('.'):
                            continue
                        item_path = os.path.join(path, item)
                        if os.path.isdir(item_path):
                            items.append((item, True))
                        elif depth < 2 and any(item.endswith(ext) for ext in ['.py', '.js', '.java', '.scala', '.go']):
                            items.append((item, False))
                
                    # Limit items to avoid overwhelming the LLM
                    for i, (item, is_dir) in enumerate(items[:max_items]):
                        if is_dir:
                            structure_lines.append(f"{indent}{item}/")
                            if depth < max_depth:
                                add_directory(os.path.join(path, item), depth + 1, 5)
                        else:
                            structure_lines.append(f"{indent}{item}")
                    
                    if len(items) > max_items:
                        structure_lines.append(f"{indent}... ({len(items) - max_items} more items)")
                        
                except PermissionError:
                    structure_lines.append(f"{indent}[Permission denied]")
            
            structure_lines.append("Repository Structure:")
            add_directory(".", 0)
            
            return "\n".join(structure_lines[:100])  # Limit total lines
            
        except Exception as e:
            get_logger().warning(f"Failed to get repository structure: {e}")
            return ""



    async def _create_issue_fix_pr(self, analysis: Dict[str, Any], files: List[str]) -> bool:
        """Create a child PR with the fix for the issue."""
        try:
            # Create unique fix branch
            timestamp = int(time.time())
            fix_branch = f"fix-issue-{self.issue.number}-{timestamp}"
            
            get_logger().info(f"Creating fix branch: {fix_branch}")
            
            # Get repository name from issue URL
            repo_path, _ = self.git_provider._parse_issue_url(self.issue_url)
            if not repo_path:
                get_logger().error("Could not parse repository from issue URL")
                return False
            
            # Get default branch for the repo
            repo_obj = self.git_provider.github_client.get_repo(repo_path)
            default_branch = repo_obj.default_branch
            
            # Create and checkout fix branch
            try:
                result = subprocess.run(["git", "checkout", "-b", fix_branch], capture_output=True, text=True, check=True)
                get_logger().info(f"Successfully created branch: {fix_branch}")
            except subprocess.CalledProcessError as e:
                get_logger().error(f"Failed to create git branch: {e.stderr}")
                return False
            
            # Generate the fix using Aider
            success = await self._generate_fix_with_aider(analysis, files)
            if not success:
                # Cleanup branch
                subprocess.run(["git", "checkout", default_branch], check=False)
                subprocess.run(["git", "branch", "-D", fix_branch], check=False)
                return False
            
            # Check if there are changes to commit
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if not result.stdout.strip():
                get_logger().warning("No changes made by fix generation")
                subprocess.run(["git", "checkout", default_branch], check=False)
                subprocess.run(["git", "branch", "-D", fix_branch], check=False)
                return False
            
            # Commit changes
            commit_message = f"🤖 Fix issue #{self.issue.number}: {analysis.get('summary', self.issue.title)}"
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            subprocess.run(["git", "push", "origin", fix_branch], check=True)
            
            # Create child PR
            pr_title = f"🔧 Fix issue #{self.issue.number}: {self.issue.title}"
            pr_body = self._create_pr_body(analysis)
            
            # Use the repo object we already have
            repo = repo_obj
            
            child_pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=fix_branch,
                base=default_branch,
                draft=True  # Start as draft
            )
            
            # Add special labels for child PR workflow
            try:
                special_labels = ["autofix-approved", "ai-generated", "child-pr"]
                for label in special_labels:
                    try:
                        child_pr.add_to_labels(label)
                        get_logger().info(f"Added label '{label}' to child PR #{child_pr.number}")
                    except Exception as label_error:
                        # Label might not exist in repo - create it or skip
                        get_logger().warning(f"Could not add label '{label}': {label_error}")
            except Exception as e:
                get_logger().warning(f"Failed to add special labels to child PR: {e}")
            
            # Comment on the issue
            issue_comment = f"""🤖 I've analyzed this issue and created a potential fix in **PR #{child_pr.number}**

[→ Review the proposed solution]({child_pr.html_url})

The fix addresses: {analysis.get('summary', 'the reported issue')}

Please review the changes and provide feedback if you'd like any adjustments!"""

            self.issue.create_comment(issue_comment)
            
            get_logger().info(f"Created child PR #{child_pr.number} for issue #{self.issue.number}")
            return True

        except Exception as e:
            get_logger().exception(f"Failed to create issue fix PR: {e}")
            return False

    async def _generate_fix_with_aider(self, analysis: Dict[str, Any], files: List[str]) -> bool:
        """Generate fix using Aider CLI - mirrors the working pr_fix.py implementation."""
        try:
            import shutil
            
            # Find Aider executable (same as pr_fix.py)
            aider_exe = os.environ.get("AIDER_EXECUTABLE")
            if not aider_exe:
                aider_exe = shutil.which("aider")
            
            if not aider_exe or not os.path.exists(aider_exe):
                get_logger().error("Aider CLI not found")
                return False
            
            # Build concise instruction (like pr_fix.py does)
            issue_instruction = f"Add missing functionality to fix issue: {self.issue.title}"
            if analysis.get('summary'):
                issue_instruction = analysis['summary']
            
            # Use concise message format like the working implementation  
            specific_instruction = f"""Fix the following issue:

{issue_instruction}

Make minimal, focused changes while ensuring existing code maintains its functionality. Follow existing code patterns and conventions."""
            
            # Prepare environment for Aider (CRITICAL - same as pr_fix.py)
            env = os.environ.copy()
            if "OPENAI_API_KEY" not in env and env.get("OPENAI_KEY"):
                env["OPENAI_API_KEY"] = env["OPENAI_KEY"]
            if "ANTHROPIC_API_KEY" not in env and env.get("ANTHROPIC_KEY"):
                env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_KEY"]
            
            # Mirror exact command structure from pr_fix.py
            msg_arg = ["--message", specific_instruction]
            cmd = [aider_exe, "--yes", "--architect", "--no-auto-commits"] + msg_arg + files
            
            get_logger().info(f"Running aider command: {' '.join(cmd)}")
            get_logger().info(f"Aider instruction: {specific_instruction[:200]}...")
            get_logger().info(f"Files to fix: {files}")
            
            # Run without timeout (like pr_fix.py) and with proper environment
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            get_logger().info(f"Aider exit code: {proc.returncode}")
            
            # Log output (same as pr_fix.py)
            get_logger().info("=== FULL AIDER STDOUT START ===")
            get_logger().info(proc.stdout)
            get_logger().info("=== FULL AIDER STDOUT END ===")
            
            if proc.stderr:
                get_logger().info("=== FULL AIDER STDERR START ===")
                get_logger().info(proc.stderr)
                get_logger().info("=== FULL AIDER STDERR END ===")
            
            if proc.returncode != 0:
                get_logger().error(f"Aider failed: {proc.stderr or proc.stdout}")
                return False
            
            get_logger().info("Aider completed successfully")
            return True

        except Exception as e:
            get_logger().exception(f"Aider execution failed: {e}")
            return False

    def _create_pr_body(self, analysis: Dict[str, Any]) -> str:
        """Create PR description for the issue fix."""
        return f"""## 🤖 AI-Generated Fix for Issue #{self.issue.number}

### What This Fixes
{analysis.get('summary', 'Addresses the reported issue')}

### Issue Type
{analysis.get('issue_type', 'bug_fix')}

### Confidence Score
{analysis.get('confidence', 0.7):.1%}

### Original Issue
> {self.issue.title}

### 🔄 Child PR Workflow
This is an **AI-generated Child PR** with special workflow capabilities:

**Need improvements?** 
- Comment `/fix` to refine the solution automatically
- Comment `/review` for detailed code analysis  
- Comment `/improve` for suggested enhancements

**Ready to merge?**
- Remove the `draft` status when satisfied
- Normal approval process applies for final merge

### How to Review & Iterate
1. 📝 **Review the changes** line by line using GitHub's review interface
2. 💬 **Comment on issues you find** (e.g., "Calculator class is missing", "Logic error in line 45")
3. 🤖 **Comment `/fix`** to have AI automatically address your feedback
4. 🔄 **Repeat steps 2-3** until satisfied with the solution
5. ✅ **Remove draft status** when ready for final approval

### Testing
Please test the changes to ensure they resolve the issue without introducing regressions.

---
🤖 *Generated by FixItFast AI*  
Closes #{self.issue.number}
"""

    async def _chat_analysis(self, model, prompt):
        """Chat with AI for issue analysis."""
        return await self.ai_handler.chat_completion(
            model=model,
            temperature=0.2,
            system="You are an expert code analyst. Analyze GitHub issues and determine if they can be fixed with code changes. Always respond with valid JSON.",
            user=prompt
        )