# ERA V3 Session 7 Assignment

## Problem Statement

1. Your new target is:  
    1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
    2. Less than or equal to 15 Epochs
    3. Less than 8000 Parameters
    4. Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.
    5. Once done, train on SageMaker or EC2 
2. Do this in exactly 3 steps (Excluding SageMaker/EC2 part)
3. Each File must have a "target, result, analysis" TEXT block (either at the start or the end)  
4. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct.  
5. Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
6. Explain your 3 steps using these targets, results, and analysis with links to your GitHub files (Colab files moved to GitHub).  
7. Keep Receptive field calculations handy for each of your models.  
8. If your GitHub folder structure or file_names are messy, -100.  
9. When ready, attempt SESSION 7 -Assignment Solution  


## Solution, Step 1 [Notebook](./ERAV3_S7_v1.ipynb)

### Target   
- Create a Setup (dataset, data loader, train/test steps and log plots)  
- Defining simple model with Convolution block, GAP, dropout and batch normalization.

### Results
- Parameters: 6,379,786
- Best Train Accuracy 99.88%  
- Best Test Accuracy 99.2%  

### Analysis
- Model with 6K parameters is able to reach till 99.2% accuracy in 15 epochs.
- Model is not overfitting as training and test accuracies are closeby.

## Solution, Step 2 [Notebook](./ERAV3_S7_v2.ipynb)

### Target   
- Add image augmentation w random rotation and random affine to improve the model performance.

### Results
- Parameters: 29,370
- Best Train Accuracy 99.85%  
- Best Test Accuracy 99.09%  

### Analysis
- Model with 6K parameters is able to reach till 99.19% accuracy in 15 epochs.
- Image augmentation doesn't show much improvement. It may be because of presense of dropout which effectively does similar function.

## Solution, Step 3 [Notebook](./ERAV3_S7_v3.ipynb)

### Target   
- Study effect of including StepLR rate scheduler.
- Increase model capacity by increasing number of convolution layer.
- Optimize the learning rate and drop out value

### Results
- Parameters: 7,760
- Best Train Accuracy 99.35%  
- Best Test Accuracy 99.43%  
### Analysis
- Model with 7.7K parameters is cross 99.43% accuracy in 15 epochs.
- Model meets all the requirement of model size, accuracy and epoch.
- Increasing model capacity and LR rate scheduler helps meet the accuracy in 15 epochs
