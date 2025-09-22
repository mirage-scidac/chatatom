#!/bin/bash

# Git Push Workflow - Intelligent commit and push automation
# Generates descriptive commit messages based on actual changes made

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to analyze changes and generate commit message
generate_commit_message() {
    local changes=$(git diff --cached --name-status)
    local new_files=$(echo "$changes" | grep "^A" | wc -l | tr -d ' ')
    local modified_files=$(echo "$changes" | grep "^M" | wc -l | tr -d ' ')
    local deleted_files=$(echo "$changes" | grep "^D" | wc -l | tr -d ' ')
    local renamed_files=$(echo "$changes" | grep "^R" | wc -l | tr -d ' ')
    
    # Default values
    local commit_type="chore"
    local scope=""
    local description=""
    
    # Analyze file patterns to determine type and scope
    if echo "$changes" | grep -q "\.md$\|\.rst$\|\.txt$\|docs/"; then
        commit_type="docs"
        scope="documentation"
        if [ $new_files -gt 0 ]; then
            description="add new documentation"
        else
            description="update documentation"
        fi
    elif echo "$changes" | grep -q "test.*\.py$\|.*test\.py$\|tests/\|spec/"; then
        commit_type="test"
        scope="testing"
        if [ $new_files -gt 0 ]; then
            description="add new test cases"
        else
            description="update existing tests"
        fi
    elif echo "$changes" | grep -q "\.py$\|\.js$\|\.ts$\|\.java$\|\.cpp$\|\.c$\|\.h$\|\.go$\|\.rs$"; then
        scope="core"
        if [ $new_files -gt 0 ] && [ $modified_files -eq 0 ]; then
            commit_type="feat"
            description="add new functionality"
        elif [ $modified_files -gt 0 ] && [ $new_files -eq 0 ]; then
            # Analyze the actual changes to determine if it's a fix or feature
            local diff_content=$(git diff --cached)
            if echo "$diff_content" | grep -q "fix\|bug\|error\|issue\|problem"; then
                commit_type="fix"
                description="resolve issues in existing code"
            elif echo "$diff_content" | grep -q "refactor\|cleanup\|reorganize"; then
                commit_type="refactor"
                description="improve code structure"
            else
                commit_type="feat"
                description="enhance existing functionality"
            fi
        else
            commit_type="feat"
            description="add new features and update existing code"
        fi
    elif echo "$changes" | grep -q "package\.json\|requirements\.txt\|Cargo\.toml\|go\.mod\|pom\.xml\|build\.gradle"; then
        commit_type="build"
        scope="dependencies"
        description="update project dependencies"
    elif echo "$changes" | grep -q "\.yml$\|\.yaml$\|Dockerfile\|\.github/\|\.gitlab-ci"; then
        commit_type="ci"
        scope="config"
        description="update CI/CD configuration"
    elif echo "$changes" | grep -q "\.css$\|\.scss$\|\.sass$\|\.less$"; then
        commit_type="style"
        scope="ui"
        description="update styling and appearance"
    elif echo "$changes" | grep -q "\.html$\|\.jsx$\|\.tsx$\|\.vue$"; then
        commit_type="feat"
        scope="ui"
        description="update user interface"
    elif echo "$changes" | grep -q "\.sh$\|Makefile\|\.bat$"; then
        commit_type="build"
        scope="scripts"
        description="update build and automation scripts"
    elif echo "$changes" | grep -q "\.gitignore\|\.env\|config\|settings"; then
        commit_type="chore"
        scope="config"
        description="update project configuration"
    fi
    
    # Handle deletions
    if [ $deleted_files -gt 0 ] && [ $new_files -eq 0 ] && [ $modified_files -eq 0 ]; then
        commit_type="refactor"
        description="remove obsolete code and files"
    fi
    
    # Handle renames
    if [ $renamed_files -gt 0 ]; then
        commit_type="refactor"
        description="reorganize and rename files"
    fi
    
    # Generate more specific descriptions based on file counts
    if [ $new_files -gt 0 ] && [ $modified_files -gt 0 ] && [ $deleted_files -gt 0 ]; then
        description="comprehensive code changes with additions, updates, and removals"
    elif [ $new_files -gt 0 ] && [ $modified_files -gt 0 ]; then
        description="add new features and update existing functionality"
    elif [ $new_files -gt 0 ] && [ $deleted_files -gt 0 ]; then
        description="replace old implementation with new code"
    elif [ $modified_files -gt 0 ] && [ $deleted_files -gt 0 ]; then
        description="update code and remove obsolete parts"
    fi
    
    # Construct commit message
    if [ -n "$scope" ]; then
        echo "${commit_type}(${scope}): ${description}"
    else
        echo "${commit_type}: ${description}"
    fi
}

# Function to get detailed change summary
get_change_summary() {
    local changes=$(git diff --cached --name-status)
    local summary=""
    
    local new_count=$(echo "$changes" | grep "^A" | wc -l | tr -d ' ')
    local mod_count=$(echo "$changes" | grep "^M" | wc -l | tr -d ' ')
    local del_count=$(echo "$changes" | grep "^D" | wc -l | tr -d ' ')
    local ren_count=$(echo "$changes" | grep "^R" | wc -l | tr -d ' ')
    
    if [ $new_count -gt 0 ]; then
        summary="${summary}${new_count} new file(s), "
    fi
    if [ $mod_count -gt 0 ]; then
        summary="${summary}${mod_count} modified file(s), "
    fi
    if [ $del_count -gt 0 ]; then
        summary="${summary}${del_count} deleted file(s), "
    fi
    if [ $ren_count -gt 0 ]; then
        summary="${summary}${ren_count} renamed file(s), "
    fi
    
    # Remove trailing comma and space
    summary=$(echo "$summary" | sed 's/, $//')
    echo "$summary"
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_status $RED "❌ Error: Not a git repository. Please run 'git init' first."
        exit 1
    fi
}

# Function to check for uncommitted changes
check_for_changes() {
    if [ -z "$(git status --porcelain)" ]; then
        print_status $GREEN "✅ No changes to commit"
        return 1
    fi
    return 0
}

# Function to show what will be committed
show_changes() {
    print_status $CYAN "📋 Changes to be committed:"
    git status --short
    echo
    
    local summary=$(get_change_summary)
    if [ -n "$summary" ]; then
        print_status $BLUE "📊 Summary: $summary"
        echo
    fi
}

# Function to handle upstream branch setup
setup_upstream() {
    local branch=$(git branch --show-current)
    local remote="origin"
    
    # Check if remote exists
    if ! git remote | grep -q "^${remote}$"; then
        print_status $YELLOW "⚠️  No remote named '$remote' found. Please add a remote first:"
        print_status $YELLOW "   git remote add origin <repository-url>"
        return 1
    fi
    
    # Check if remote branch exists
    if ! git ls-remote --heads $remote $branch 2>/dev/null | grep -q $branch; then
        print_status $YELLOW "🌿 Setting upstream branch for '$branch'..."
        git push --set-upstream $remote $branch
        return $?
    fi
    
    return 0
}

# Function to perform the push
do_push() {
    local branch=$(git branch --show-current)
    local remote="origin"
    
    print_status $BLUE "🚀 Pushing to $remote/$branch..."
    
    # Try normal push first
    if git push 2>/dev/null; then
        return 0
    fi
    
    # If normal push fails, try setting upstream
    if setup_upstream; then
        return 0
    fi
    
    return 1
}

# Function to perform intelligent push workflow
intelligent_push() {
    local custom_message=""
    local force_push=false
    local dry_run=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--message)
                custom_message="$2"
                shift 2
                ;;
            -f|--force)
                force_push=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_status $RED "❌ Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_status $PURPLE "🔍 Analyzing changes..."
    
    # Check if there are any changes
    if ! check_for_changes; then
        return 0
    fi
    
    # Show what will be committed
    show_changes
    
    # Stage all changes
    print_status $YELLOW "📦 Staging changes..."
    git add .
    
    # Generate or use custom commit message
    local commit_msg
    if [ -n "$custom_message" ]; then
        commit_msg="$custom_message"
        print_status $CYAN "💬 Using custom commit message: $commit_msg"
    else
        commit_msg=$(generate_commit_message)
        print_status $CYAN "💬 Generated commit message: $commit_msg"
    fi
    
    # Dry run mode
    if [ "$dry_run" = true ]; then
        print_status $YELLOW "🧪 Dry run mode - would commit with message: $commit_msg"
        return 0
    fi
    
    # Commit changes
    print_status $GREEN "💾 Committing changes..."
    if ! git commit -m "$commit_msg"; then
        print_status $RED "❌ Failed to commit changes"
        return 1
    fi
    
    # Push changes
    if [ "$force_push" = true ]; then
        print_status $YELLOW "⚠️  Force pushing changes..."
        git push --force-with-lease
    else
        if ! do_push; then
            print_status $RED "❌ Failed to push changes"
            return 1
        fi
    fi
    
    print_status $GREEN "✅ Successfully pushed changes!"
    
    # Show final status
    local branch=$(git branch --show-current)
    local commit_hash=$(git rev-parse --short HEAD)
    print_status $BLUE "📍 Latest commit: $commit_hash on branch '$branch'"
}

# Function to show help
show_help() {
    cat << EOF
Git Push Workflow - Intelligent commit and push automation

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -m, --message <msg>    Use custom commit message instead of auto-generated
    -f, --force           Force push with --force-with-lease
    --dry-run             Show what would be done without actually doing it
    -h, --help            Show this help message

EXAMPLES:
    $0                                    # Auto-generate commit message and push
    $0 -m "feat: add new feature"         # Use custom commit message
    $0 --dry-run                          # Preview what would be committed
    $0 -f                                 # Force push changes

COMMIT MESSAGE FORMAT:
    <type>(<scope>): <description>
    
    Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build
    
EOF
}

# Main execution
main() {
    # Check if we're in a git repository
    check_git_repo
    
    # Run the intelligent push workflow
    intelligent_push "$@"
}

# Execute main function with all arguments
main "$@"
