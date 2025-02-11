# Public version of the MLBB-AI-Drafter Repo

> 💡 Note: This webapp requires backend requires to function properly. If you'd like to see the fully functioning version of this app, please contact me via email at jasonavilasoria@gmail.com.

Here's a quick look at what the app does (however, if you'd like to run the frontend yourself see [this](#no-access)):

1. Loads models all trained locally. Model architectures include:
    - Deep LSTM models
    - Enchanced Convid LSTM models
    - Parallel Convid GRU models
    - Deep Convid models
    - & Many more, including "Bigger" versions of each one above for a total of 18 <strong> Unique Models!</strong>
2. Once loaded, hero data is given to the frontend and looks like this when the form is populated:
    - ![MainImage](/img/main.png) 
3. The populated dropdowns look as such:
    - ![PopImage](/img/pop.png)
4. Here, you are free to choose your draft. Familiarity with `Mobile Legends` is important but not necessary.
    - Banned Heros will *NOT* be part of the prediction
5. After choosing your draft, enter how many reccomdations you'd like. This will change the prediction count and be easier on your system (if running locally).
    - ![Main2Image](/img/main2.png)
6. This image is who the models think **you** should pick next. Every hero changes predictions! 
    - ![Main3Image](/img/main3.png)
7. This image is who the models think **the enemy** will pick next. Every hero changes predictions! 
    - ![Main4Image](/img/main4.png)
8. Finally, If you disagree, these models are based off of community feedback, further reinforcing the picks. The suggestions are stored and models are trained on these suggestions weekly
    - ![Main5Image](/img/main5.png)


## Access

If you've been given access to help me out developing this app, please see the `README` in the [Backend](/backend/README.MD) folder!

## No access

If no access to the models or backend has been granted, you can use a limited version of the app by running a `development build` of Node. Here's how:
* Install the latest version of Node.
1. Run `cd frontend`
2. Run `npm install`
3. Run `npm run dev`

Remember, some options will not be populated and you will mostly see the frontend with little functionality.