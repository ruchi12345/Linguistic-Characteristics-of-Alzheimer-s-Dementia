# Linguistic-Characteristics-of-Alzheimer-s-Dementia

In this repo, I have used the dataset which I have provided inthe dataset folder. you can download the datset from here. https://dementia.talkbank.org/

Dataset shall be taken in some control environment nd all the reponses should be written in txt.

In the current repo, The following dataset has been used and preprocessed it and make a csv to increase the accuracy. 
The increased data will further manually cleaned by me.

The preprocessing script can be found in preprocessing folder and preprocessed output csv can be found in this repo as well.
I have applied many machine learning algorithms and obtain maximum accuracy with classication models.

The overall limitation of data even after processing and manually cleaned is unballanced.
Then I applied many resampling techniques.
Now, Its time to again apply classification algorithms as the sie of data become high. I have other options to try as well. I moved to deep learning.

The first model I apply was using Dense layer and obtained less accuracy.
Now its time to rethink for solution.
So, I build deeper network with Hybrid model. My Deep Network looks like this.You can find the png here.








By seeing this PNG, I get horrified. I Hope I get good accuracy now. But still the accuracy is low.

Now its time to better go for Re-solution.
Either, I go back to resample and check out some more ML algorithms. Thinking about resampling is time consuming as i am a datascientist. I prefer to go with checking more classification algorithm. We have achieved 93% accuracy with my all time favorite random forest Model.
So this is how ML can beat DL.







