# AdvanceAlgoGroup4

SortingAlgorithms

## Instructions to run the Application ( MacOS )

1. Clone the repository using this URL

```bash
git clone https://github.com/keshavlingala/AlgoAnalyzerTool
cd AlgoAnalyzerTool
chmod +x setup app # This will give execute permission to setup and app files
./setup # This create a virtual environment and install all the dependencies
./app # This will run the application
```

## Rules

* All changes must be made using a Pull Request (PR), Follow instructions below to do one

## Steps to create a PR and merge your changes

1. Clone the repository using this URL

```bash
git clone https://github.com/keshavlingala/AlgoAnalyzerTool
```

2. Make a branch/s for your changes, use the below command to create a branch

```bash
git checkout -b <your-name>
```

> Example
> ```bash
> git checkout -b keshav
> ```

3. Make your changes...
4. Add your changes using the below command

```bash
git add filename.py
```

5. Commit Your Changes

```bash
git commit -m "description of changes"
```

6. Push your changes

```bash
git push origin <branch-name>
```

> Example
> ```bash
> git push origin keshav
> ```

7. This will create a branch in the github.com (remote)

8. Now go to this URL https://github.com/keshavlingala/AlgoAnalyzerTool/branches here, you can see your branch , on your
   branch name, click `New pull request` button

9. Add title and description(optional) for the PR and click `Create pull request`

8. And then copy the URL and paste in our group chat to request for code review.

9. After at least 1 approval, you can merge your branch using the `Merge Pull Request` in the same PR link.

> Important

10. When your changes are merged and you are going to make new changes `YOU MUST FOLLOW BELOW STEPS`

11. Rebase your branch to main branch before making new changes

```bash
git pull origin master
git rebase master
```



