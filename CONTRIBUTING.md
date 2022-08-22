## 🦾Thank you for contributing!

**Please follow these steps to get your work merged in.**

1. Add a [GitHub Star](https://github.com/agusle/credit-risk-analysis-using-deep-learning) to the project.
2. Clone repo and create a new branch: `$ git checkout https://github.com/agusle/credit-risk-analysis-using-deep-learning -b name_for_new_branch.`
3. Add a feature, fix a bug, or refactor some code :)
4. Write/update tests for the changes you made, if necessary.
5. Update `README.md` and `CONTRIBUTORS.md`, if necessary.
4. Submit Pull Request with comprehensive description of changes

## ⚙️ Install
You have 2 options depending on whether you want to run the application on CPU or GPU:

- **CPU:**

```bash
$ cd docker/
$ docker build -t credit_risk_analysis -f Dockerfile .
```

- **GPU:**

```bash
$ docker build -t credit_risk_analysis -f Dockerfile_gpu .
```

## ⚡️ Run 

```bash
$ docker run --rm -it \
    -p 8889:8889 \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    credit-risk-analysis \
    bash
```

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

```bash
$ jupyter notebook --ip 0.0.0.0 --port 8889 --allow-root
```