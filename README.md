```
Root
└── Data 
│   ├──  (all datasets go to this folder)
└── Pipeline
│   ├──  DialogueStateManager.py
│   ├──  EmotionClassifier.py
│   ├──  IntentClassifier.py
│   └──  ResponseGenerator.py
├── models
│   ├── emotions
│   │	└── Emotion classification model
│   └── intents
│   │	└── Intent classification model
│   └── gpt2
│	└── response generator model
├── pipeline.py (main of the dialogue system)
├── train_emotion.py (for training emotion classifier)
├── train_intent.py (for training intention classifier)
├── check_maxlength.py (for checking maxlength of final prompt)
└── dataset.py (dataset consolidated for all trainings)
```


1.	Install Anaconda in the machine and create environment  and make sure CUDA and GPU are properly install if y want to use GPU for finetuning or inference in pipeline.
`conda create --name <env_name> python=3.9`
2.	Download the repo from Github and all the models from cloud drive
3.	Replace all 3 paths to the model directories for 3 models in pipeline> load_models() 
4.	Replace the path to the persona dataset to get the persona to where it is stored.
5.	Go to the root directory of the system
`cd <path_to_root_dir_of_repo>`
6.	Install all necessary requirements
`pip install -r requirements.txt`
`python -m spacy download en_core_web_sm`
7.	Run the system
`python pipeline.py`
After the chatbot is prepared as show in console, start talking with the chatbot.

8.	Anytime you want to end the conversation type `exit` in the conversation to signal the system for ending the process.
