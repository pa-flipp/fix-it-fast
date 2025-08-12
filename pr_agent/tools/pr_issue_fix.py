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

            files = self._discover_relevant_files(analysis)
            if not files:
                get_logger().error("No relevant files found for issue")
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

    def _discover_relevant_files(self, analysis: Dict[str, Any]) -> List[str]:
        """Find relevant files for the issue."""
        try:
            files = []
            keywords = analysis.get('keywords', [])
            
            # Strategy 1: Look for explicit file mentions in issue
            issue_text = self.issue.title + " " + (self.issue.body or "")
            file_pattern = r'\b[\w/.-]+\.\w+\b'
            potential_files = re.findall(file_pattern, issue_text)
            
            for file_path in potential_files:
                if self._is_valid_code_file(file_path):
                    files.append(file_path)
            
            # Strategy 2: Search for files with matching keywords
            if len(files) < 3:  # Need more files
                keyword_files = self._find_files_by_keywords(keywords)
                files.extend(keyword_files)
            
            # Limit to max files and apply safety filters
            max_files = 5  # Hackathon limit
            files = files[:max_files]
            
            # Filter by allowed extensions (reuse from PR workflow)
            allowed_extensions = ['.py', '.js', '.ts', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.rb', '.php']
            filtered_files = []
            for file_path in files:
                if any(file_path.endswith(ext) for ext in allowed_extensions):
                    if os.path.exists(file_path):
                        filtered_files.append(file_path)
            
            get_logger().info(f"Found {len(filtered_files)} relevant files: {filtered_files}")
            return filtered_files

        except Exception as e:
            get_logger().exception(f"File discovery failed: {e}")
            return []

    def _is_valid_code_file(self, file_path: str) -> bool:
        """Check if file path looks like a valid code file."""
        code_extensions = ['.py', '.js', '.ts', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.rb', '.php', '.scala', '.kt']
        return any(file_path.endswith(ext) for ext in code_extensions)

    def _find_files_by_keywords(self, keywords: List[str]) -> List[str]:
        """Find files containing keywords (simple implementation)."""
        try:
            import glob
            files = []
            
            # Search for files containing keywords in their names
            for keyword in keywords:
                pattern = f"**/*{keyword}*"
                matches = glob.glob(pattern, recursive=True)
                for match in matches:
                    if self._is_valid_code_file(match) and os.path.isfile(match):
                        files.append(match)
            
            return list(set(files))  # Remove duplicates
        except Exception:
            return []

    async def _create_issue_fix_pr(self, analysis: Dict[str, Any], files: List[str]) -> bool:
        """Create a child PR with the fix for the issue."""
        try:
            # Create unique fix branch
            timestamp = int(time.time())
            fix_branch = f"fix-issue-{self.issue.number}-{timestamp}"
            
            get_logger().info(f"Creating fix branch: {fix_branch}")
            
            # Get default branch for the repo
            repo_obj = self.git_provider.github_client.get_repo(self.git_provider.repo)
            default_branch = repo_obj.default_branch
            
            # Create and checkout fix branch
            subprocess.run(["git", "checkout", "-b", fix_branch], check=True)
            
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
            
            from github import Github
            g = self.git_provider.github_client
            repo = g.get_repo(self.git_provider.repo)
            
            child_pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=fix_branch,
                base=default_branch,
                draft=True  # Start as draft
            )
            
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
        """Generate fix using Aider CLI."""
        try:
            import shutil
            
            # Find Aider executable
            aider_exe = os.environ.get("AIDER_EXECUTABLE")
            if not aider_exe:
                aider_exe = shutil.which("aider")
            
            if not aider_exe:
                get_logger().error("Aider CLI not found")
                return False
            
            # Build fix instruction
            fix_instruction = f"""
Fix this GitHub issue:

ISSUE: {self.issue.title}

DESCRIPTION:
{self.issue.body or 'No description provided'}

ANALYSIS: {analysis.get('summary', 'Address the reported issue')}

Please make minimal, focused changes to fix the issue. Focus on:
- Fixing the root cause
- Adding proper error handling
- Ensuring code safety
- Following existing code patterns

Files to consider: {', '.join(files)}
"""

            # Run Aider
            cmd = [
                aider_exe,
                "--no-auto-commits",
                "--yes",
                "--architect",
                "--message", fix_instruction
            ] + files
            
            get_logger().info(f"Running Aider: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                get_logger().info("Aider completed successfully")
                return True
            else:
                get_logger().error(f"Aider failed: {result.stderr}")
                return False

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

### How to Review
1. 📝 **Review the changes** line by line using GitHub's review interface
2. 💬 **Comment on specific lines** if you want modifications
3. 🔄 **Add general comments** for broader changes
4. ✅ **Approve when satisfied**

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