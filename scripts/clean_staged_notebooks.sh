#!/bin/bash

# Get list of staged .ipynb files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.ipynb$')

# Loop through the staged .ipynb files and clean them
for file in $STAGED_FILES; do
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$file"

    # Stage the cleaned file
    git add "$file"
done