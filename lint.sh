#!/usr/bin/bash

# remove trailing whitespace
find . -name "*.py" | xargs sed -i '' -e's/[ ^I]*$//'

# lint project
echo 'compfipy' > linting_report.txt
pylint compfipy >> linting_report.txt

echo ' ' >> linting_report.txt
echo 'tests' >> linting_report.txt
pylint tests >> linting_report.txt

echo "project linted"
