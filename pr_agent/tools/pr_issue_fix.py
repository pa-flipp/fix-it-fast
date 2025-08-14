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
                if self._is_valid_target(file_path):
                    files.append(file_path)
                    get_logger().info(f"Found explicitly mentioned file: {file_path}")
            
            # Strategy 2: LLM-powered intelligent file discovery
            llm_files = await self._discover_relevant_files_with_llm(analysis)
            if llm_files:
                files.extend(llm_files)
                get_logger().info(f"LLM discovered {len(llm_files)} relevant files")
            
            # Strategy 3: Fallback keyword-based discovery if LLM found nothing
            if not llm_files and not files:
                get_logger().warning("LLM found no files, trying fallback keyword search")
                fallback_files = await self._fallback_keyword_search(analysis, keywords)
                if fallback_files:
                    files.extend(fallback_files)
                    get_logger().info(f"Fallback search found {len(fallback_files)} files")
            
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

    def _is_valid_target(self, path: str) -> bool:
        """Check if path is a valid target (file or directory) for processing."""
        try:
            import os
            
            # Normalize path and ensure it's repo-root-relative
            path = path.strip()
            if not path or path.startswith('../') or '..' in path:
                return False
                
            # Check if path exists
            if not os.path.exists(path):
                return False
                
            # Handle directories
            if os.path.isdir(path):
                # Check if directory contains tracked files
                try:
                    import subprocess
                    result = subprocess.run(
                        ['git', 'ls-files', path], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return True
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Fallback: check if directory has code files
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            if self._is_valid_code_file(file):
                                return True
                return False
                
            # Handle files
            return self._is_valid_code_file(path)
            
        except Exception as e:
            get_logger().debug(f"Path validation failed for {path}: {e}")
            return False
    
    def _is_valid_code_file(self, file_path: str) -> bool:
        """Check if file path looks like a valid code file."""
        # Code extensions
        code_extensions = [
            '.py', '.js', '.ts', '.tsx', '.jsx',           # Web/Python
            '.java', '.scala', '.sbt', '.gradle',         # JVM languages  
            '.go', '.rs', '.cpp', '.c', '.h',             # Systems languages
            '.rb', '.php', '.swift', '.kt',               # Other languages
            '.sql', '.yaml', '.yml', '.json', '.toml'     # Config files
        ]
        
        # Check extensions
        if any(file_path.endswith(ext) for ext in code_extensions):
            return True
            
        # Extensionless allowlist
        extensionless_files = {
            'Makefile', 'Dockerfile', 'Jenkinsfile', 'Procfile', 
            'BUILD', 'WORKSPACE', 'CMakeLists.txt'
        }
        
        filename = os.path.basename(file_path)
        if filename in extensionless_files:
            return True
            
        # Dockerfile variations
        if filename.startswith('Dockerfile.'):
            return True
            
        return False



    async def _discover_relevant_files_with_llm(self, analysis: dict) -> list[str]:
        """Use LLM to intelligently discover relevant files for the issue."""
        try:
            from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
            
            ai_handler = LiteLLMAIHandler()
            
            # Build comprehensive repository context
            repo_structure = self._get_repository_structure()
            issue_context = f"Title: {self.issue.title}\nBody: {self.issue.body or 'No description'}"
            
            prompt = f"""You are a code analysis expert. Given this GitHub issue and repository structure, identify the most relevant files and directories for fixing this issue.

**Issue:**
{issue_context}

**Repository Structure:**
{repo_structure}

**Analysis Context:**
- Issue Type: {analysis.get('issue_type', 'Unknown')}
- Summary: {analysis.get('summary', 'No summary')}
- Keywords: {', '.join(analysis.get('keywords', []))}

**File Selection Instructions:**
- Return at most 10 paths total, repo-root-relative
- Each path prefixed with "file:" or "dir:" and confidence score
- No globs, wildcards, or absolute paths
- Focus on files/directories relating to: {analysis.get('summary', 'this issue')}
- Prioritize paths containing these keywords: {', '.join(analysis.get('keywords', []))}
- Prefer specific files; include directories only when they contain multiple relevant files
- Paths should be chosen to keep final expanded file count under 15

**Output Format (each line):**
file: path/to/specific.scala | 0.92 | contains performance logic
dir: scala_processing/search_module/ | 0.78 | performance-related module

**Rules:**
- Only return paths that exist in the repository structure above
- No explanations outside the format
- Higher confidence (0.0-1.0) for more relevant paths
- Include brief reason after confidence score"""

            response = await ai_handler.chat_completion(
                model="gpt-4o-mini",
                system="You are a code analysis expert. Analyze GitHub issues and repository structures to identify relevant files for fixing issues.",
                user=prompt,
                temperature=0.1
            )
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                get_logger().warning("LLM file discovery: No response from AI")
                return []
            
            content = response.choices[0].message.content.strip()
            if not content:
                get_logger().warning("LLM file discovery: Empty response from AI")
                return []
            
            # Parse structured response
            files, directories = self._parse_structured_llm_response(content)
            get_logger().info(f"LLM suggested {len(files)} files and {len(directories)} directories")
            
            # Validate and collect all targets
            valid_targets = []
            
            # Add valid files
            for file_path, confidence, reason in files:
                if self._is_valid_target(file_path):
                    valid_targets.append((file_path, confidence, reason, 'file'))
                    get_logger().info(f"Valid file: {file_path} (confidence: {confidence:.2f})")
                else:
                    get_logger().debug(f"Invalid file: {file_path}")
            
            # Add valid directories (will be expanded later)
            for dir_path, confidence, reason in directories:
                if self._is_valid_target(dir_path):
                    valid_targets.append((dir_path, confidence, reason, 'directory'))
                    get_logger().info(f"Valid directory: {dir_path} (confidence: {confidence:.2f})")
                else:
                    get_logger().debug(f"Invalid directory: {dir_path}")
            
            # Sort by confidence and expand directories
            valid_targets.sort(key=lambda x: x[1], reverse=True)
            final_files = await self._expand_targets_to_files(valid_targets, analysis)
            
            return final_files[:15]  # Final limit
            
        except Exception as e:
            get_logger().exception(f"LLM file discovery failed: {e}")
            return []

    def _parse_structured_llm_response(self, response_text: str) -> tuple[list, list]:
        """Parse structured LLM response with confidence scores."""
        files, directories = [], []
        
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            try:
                if line.startswith('file:'):
                    # Parse: file: path/to/file.py | 0.85 | reason
                    parts = [p.strip() for p in line.split(' | ')]
                    path = parts[0][5:].strip()  # Remove 'file:'
                    confidence = float(parts[1]) if len(parts) > 1 else 0.5
                    reason = parts[2] if len(parts) > 2 else "AI suggested"
                    files.append((path, confidence, reason))
                    
                elif line.startswith('dir:'):
                    # Parse: dir: path/to/directory | 0.75 | reason
                    parts = [p.strip() for p in line.split(' | ')]
                    path = parts[0][4:].strip()  # Remove 'dir:'
                    confidence = float(parts[1]) if len(parts) > 1 else 0.5
                    reason = parts[2] if len(parts) > 2 else "AI suggested"
                    directories.append((path, confidence, reason))
                    
            except (ValueError, IndexError) as e:
                get_logger().debug(f"Failed to parse LLM response line: {line} - {e}")
                continue
                
        return files, directories

    async def _expand_targets_to_files(self, targets: list, analysis: dict) -> list[str]:
        """Expand directories to files and return final file list."""
        try:
            final_files = []
            keywords = [kw.lower() for kw in analysis.get('keywords', [])]
            
            for path, _confidence, _reason, target_type in targets:
                if target_type == 'file':
                    # Direct file - add immediately
                    final_files.append(path)
                    get_logger().info(f"Added direct file: {path}")
                    
                elif target_type == 'directory':
                    # Expand directory to files
                    dir_files = self._expand_directory_to_files(path, keywords, max_files=5)
                    final_files.extend(dir_files)
                    get_logger().info(f"Expanded directory {path} to {len(dir_files)} files")
                    
                # Stop if we've reached our limit
                if len(final_files) >= 15:
                    break
                    
            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for f in final_files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
                    
            return unique_files[:15]
            
        except Exception as e:
            get_logger().exception(f"Target expansion failed: {e}")
            return []

    def _expand_directory_to_files(self, directory: str, keywords: list[str], max_files: int = 5) -> list[str]:
        """Expand a single directory to relevant files using git and keyword matching."""
        try:
            import subprocess
            import os
            
            # First try git ls-files for tracked files
            try:
                result = subprocess.run(
                    ['git', 'ls-files', directory],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    tracked_files = result.stdout.strip().split('\n')
                    tracked_files = [f for f in tracked_files if f.strip()]
                else:
                    tracked_files = []
                    
            except (subprocess.SubprocessError, FileNotFoundError):
                tracked_files = []
                
            # Fallback to os.walk if git failed
            if not tracked_files:
                for root, dirs, filenames in os.walk(directory):
                    # Skip common build/cache directories
                    dirs[:] = [d for d in dirs if d not in {
                        'target', 'build', 'node_modules', '.git', '.idea', 
                        '.metals', '.bloop', '.cache', 'dist', '__pycache__'
                    }]
                    
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        if self._is_valid_code_file(file_path):
                            tracked_files.append(file_path)
            
            # Filter and score files
            scored_files = []
            for file_path in tracked_files:
                if not self._is_valid_code_file(file_path):
                    continue
                    
                # Skip large files
                try:
                    if os.path.getsize(file_path) > 500 * 1024:  # 500KB limit
                        continue
                except OSError:
                    continue
                    
                # Calculate relevance score
                score = self._calculate_file_relevance_score(file_path, keywords)
                scored_files.append((file_path, score))
                
            # Sort by score and return top files
            scored_files.sort(key=lambda x: x[1], reverse=True)
            result_files = [f[0] for f in scored_files[:max_files]]
            
            get_logger().info(f"Directory {directory}: found {len(tracked_files)} files, returning {len(result_files)}")
            return result_files
            
        except Exception as e:
            get_logger().exception(f"Directory expansion failed for {directory}: {e}")
            return []

    def _calculate_file_relevance_score(self, file_path: str, keywords: list[str]) -> float:
        """Calculate relevance score for a file based on keywords."""
        score = 0.0
        
        # Normalize file path for matching
        normalized_path = file_path.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
        
        # Keyword matching in path/filename
        for keyword in keywords:
            if keyword in normalized_path:
                score += 1.0
                
        # Boost for certain file types based on common patterns
        if any(ext in file_path for ext in ['.scala', '.java']):
            score += 0.2
        if 'test' in file_path and 'test' not in [kw.lower() for kw in keywords]:
            score -= 0.5  # De-prioritize tests unless explicitly mentioned
        if any(pattern in file_path for pattern in ['generated', '.min.', 'target/', 'build/']):
            score -= 1.0  # De-prioritize generated/build files
            
        # Boost for main source files
        if 'src/main' in file_path:
            score += 0.3
            
        return max(0.0, score)  # Ensure non-negative

    async def _fallback_keyword_search(self, analysis: dict, keywords: list[str]) -> list[str]:
        """Fallback search using git grep and directory scanning when LLM fails."""
        try:
            import subprocess
            
            fallback_files = []
            issue_keywords = analysis.get('keywords', []) + keywords
            
            # Try git grep for each keyword
            for keyword in issue_keywords[:3]:  # Limit to top 3 keywords
                try:
                    result = subprocess.run(
                        ['git', 'grep', '-l', '--fixed-strings', keyword],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        grep_files = result.stdout.strip().split('\n')
                        for file_path in grep_files[:5]:  # Limit per keyword
                            if file_path and self._is_valid_target(file_path):
                                fallback_files.append(file_path)
                                get_logger().info(f"Git grep found: {file_path} (keyword: {keyword})")
                                
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
            
            # If still no files, do a broader directory scan
            if not fallback_files:
                get_logger().info("Git grep failed, trying directory scan")
                common_dirs = ['src', 'lib', 'app', 'main', 'scala_processing', 'python_processing']
                
                for dir_name in common_dirs:
                    if os.path.exists(dir_name) and os.path.isdir(dir_name):
                        dir_files = self._expand_directory_to_files(dir_name, issue_keywords, max_files=3)
                        fallback_files.extend(dir_files)
                        if len(fallback_files) >= 10:  # Don't overwhelm
                            break
            
            return fallback_files[:10]  # Final limit
            
        except Exception as e:
            get_logger().exception(f"Fallback keyword search failed: {e}")
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
                    for _i, (item, is_dir) in enumerate(items[:max_items]):
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