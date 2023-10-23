#!/bin/bash

# Checking if commit message is supplied
if [ -z "$1" ]
then
    echo "No commit message provided, exiting."
    exit 1
fi

echo "==== Current Working Branch ===="
# Check the current working branch
currentBranch=$(git rev-parse --abbrev-ref HEAD)
echo "Current working branch: $currentBranch"

echo "==== Git Status Before Changes are Added ===="
# Print git status
git status

echo "==== Changes in Files ===="
# List changed files
git diff --name-only

echo "==== Adding Changes to Staging Area ===="
# Add all changes to the staging area
git add .

echo "==== Committing Changes ===="
# Commit the changes
commit_result=$(git commit -m "$1")
if [[ $commit_result == *"nothing to commit"* ]]; then
    echo -e "\033[0;33mAlert! No changes detected in the files, nothing to commit or push.\033[0m"   # Yellow
    exit 1
fi

echo "==== Details of Latest Commit ===="
# Print the most recent commit
git log -1 --pretty=format:"%h%x09%an%x09%ad%x09%s"

echo "==== Pushing Changes to '$currentBranch' Branch ===="
# Push the changes to the current branch
git push origin $currentBranch
push_exit_status=$?  # save the exit status of the git push command

# if the push was not successful, try to pull the latest changes
if [[ $push_exit_status -ne 0 ]]; then
    echo "Push failed, trying to pull the latest changes..."
    git pull origin $currentBranch
    # after pulling the latest changes, try pushing again
    git push origin $currentBranch
    push_exit_status=$?  # save the exit status again
    # check if the push was successful this time
    if [[ $push_exit_status -ne 0 ]]; then
        echo -e "\033[0;31mError occurred! Could not push the changes even after pulling the latest changes. Please resolve conflicts manually.\033[0m"  # Red
        exit 1
    fi
fi

echo "==== Git Status After Push ===="
# Print git status after push
git status

echo "==== Log of Last 5 Commits ===="
# Print the log of the latest 5 commits 
git log --pretty=format:"%h%x09%an%x09%ad%x09%s" -5

echo "==== Verifying Everything Worked as Planned ===="
uncommitted_changes=$(git status --porcelain)
if [[ -z "$uncommitted_changes" ]]; then
    echo -e "\033[0;32mAll changes were successfully committed and pushed!\033[0m"     # Green
else
    echo -e "\033[0;31mError occurred! There are uncommitted changes. Process did not complete successfully.\033[0m"  # Red
fi

# Print the latest commit hash
latest_commit=$(git rev-parse --short HEAD)
echo "Latest commit hash: $latest_commit"

# Print the current local time
current_time=$(date)
echo "Current local time: $current_time"
echo "==== Time Since Last Save ===="

current_time=$(date +%s)                # Current timestamp in seconds
last_saved_time=$(git show -s --format=%ct HEAD)   # Timestamp of the last commit
time_diff=$((current_time - last_saved_time))      # Difference in timestamps

# Calculate days, hours, minutes and seconds
days=$((time_diff/(60*60*24)))
hours=$((time_diff/(60*60)%24))
minutes=$((time_diff/60%60))
seconds=$((time_diff%60))

if [[ $days -gt 0 ]]; then
    echo "Time since last save: $days days, $hours hours, $minutes minutes, $seconds seconds."
elif [[ $hours -gt 0 ]]; then
    echo "Time since last save: $hours hours, $minutes minutes, $seconds seconds."
elif [[ $minutes -gt 0 ]]; then
    echo "Time since last save: $minutes minutes, $seconds seconds."
else
    echo "Time since last save: $seconds seconds."
fi