#!/bin/bash

# Git Push Workflow Setup Script
# Sets up the intelligent Git push workflow for this project

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_status $BLUE "🚀 Setting up Git Push Workflow..."

# Make the main script executable
if [ -f "push_workflow.sh" ]; then
    chmod +x push_workflow.sh
    print_status $GREEN "✅ Made push_workflow.sh executable"
else
    print_status $YELLOW "⚠️  push_workflow.sh not found in current directory"
fi

# Create useful aliases
print_status $BLUE "📝 Creating helpful aliases..."

cat << 'EOF' > .git_aliases
# Git Push Workflow Aliases
# Source this file in your shell configuration (.bashrc, .zshrc, etc.)

alias gp='./push_workflow.sh'
alias gpd='./push_workflow.sh --dry-run'
alias gpf='./push_workflow.sh --force'
alias gpm='./push_workflow.sh -m'
alias gph='./push_workflow.sh --help'

# Traditional git aliases for comparison
alias gps='git add . && git commit && git push'
alias gst='git status'
alias glog='git log --oneline -10'
EOF

print_status $GREEN "✅ Created .git_aliases file"

# Create VS Code task configuration
if [ -d ".vscode" ] || [ -f ".vscode/settings.json" ]; then
    mkdir -p .vscode
    
    cat << 'EOF' > .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Git Push Workflow",
            "type": "shell",
            "command": "./push_workflow.sh",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": []
        },
        {
            "label": "Git Push Workflow (Dry Run)",
            "type": "shell",
            "command": "./push_workflow.sh",
            "args": ["--dry-run"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": []
        }
    ]
}
EOF
    
    print_status $GREEN "✅ Created VS Code tasks configuration"
fi

# Set up Git hooks (optional)
read -p "Do you want to set up a pre-push hook for validation? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p .git/hooks
    
    cat << 'EOF' > .git/hooks/pre-push
#!/bin/bash
# Pre-push hook to validate changes
echo "🔍 Validating changes before push..."
./push_workflow.sh --dry-run
EOF
    
    chmod +x .git/hooks/pre-push
    print_status $GREEN "✅ Created pre-push validation hook"
fi

print_status $BLUE "📚 Setup complete! Here's how to use the workflow:"
echo
echo "Basic usage:"
echo "  ./push_workflow.sh                 # Auto-commit and push"
echo "  ./push_workflow.sh --dry-run       # Preview changes"
echo "  ./push_workflow.sh -m 'message'    # Custom message"
echo
echo "To use aliases, add this to your shell config:"
echo "  source $(pwd)/.git_aliases"
echo
print_status $GREEN "🎉 Git Push Workflow is ready to use!"
