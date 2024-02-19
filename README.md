# Import a GPT neo Model with Inferless using docker 

## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Docker**. You would docker to build and test the container locally
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.


## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Docker(custom code)** as your model source and then choose Dockerfile with GIT as the provider and use the forked repo URL as the **Model URL**.

Enter the the Health API 

```
/healthcheck
```

Enter the the Infer API 
```
/generate
```

Enter the the Port
```
7000
```

### Input
```json
{
  "prompt":  "Once upon a time ", 
  "max_length": 50
}
```

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/docker) for more information on model import.

The following is a sample Input and Output JSON for this model which you can use while importing this model on Inferless.
