# Manual for the sphinx documentation pages



### Change directory to [docs/sphinx](docs/sphinx).

### Modify/add files in [docs/sphinx/source](docs/sphinx/source)

### In the OS console, run:
```console
make clean
make html
```

### Copy all files under [docs/sphinx/build/html](docs/sphinx/build/html) to [docs](docs).

**Note: Replace all files under [docs](docs).**

**Keep `.nojekyll` if you need to clean the `docs` folder.**



### For windows:

```shell
.\make.bat clean
.\make.bat html
```



