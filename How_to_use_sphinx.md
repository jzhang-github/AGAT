# Manual for the sphinx documentation pages

1. Change directory to [docs/sphinx](docs/sphinx).

2. Modify/add files in [docs/sphinx/source](docs/sphinx/source)

3. generate html files.

   For the first run:
   
   - ```shell
     sphinx-quickstart.exe
     ```
   
   - ```shell
     sphinx-build.exe source build
     ```
   
   Next time, run:
   
   ```shell
   .\make.bat clean
   .\make.bat html
   ```
   
4. Copy all files under [docs/sphinx/build/html](docs/sphinx/build/html) to [docs](docs).



## Note:

1. Replace all files under [docs](docs).
2. Keep `.nojekyll` if you need to clean the `docs` folder.



## Sphinx style basics

https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

Go to OneNote for more details.
