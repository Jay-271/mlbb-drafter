# Main backend folder

* To init, using python 3.11.x run `python3 -m venv work` in this folder. The `.gitignore` has this folder entry already omitted as to not push the venv out
* Do `.\work\scripts\activate` for windows or use `source` command if on Mac
* Once in the venv, run `pip install -r requirements.txt`
* Finally, run `python3 app.py`. You should see the website at `localhost:5000`

Great! Now we can serve the backend **assuming models have been shared**. Note that without the models, these steps will not work.