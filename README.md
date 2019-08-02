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




![Image description](https://agora-file-storage-prod.s3-us-west-1.amazonaws.com/workplace/attachment/5247000078979398411?response-content-disposition=inline%3B%20filename%3D%22mBdel.png%22%3B%20filename%2A%3Dutf-8%27%27mBdel.png&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEFwaCXVzLXdlc3QtMSJGMEQCIDjnryZ1bAp7FdG19a8lNq1p7Hkkdd2YDT%2FS0xjF1gNbAiBIamH%2B30ENxtgIRPDKccW6qLNGCgRgjAAr%2FQQUWMyw9yrjAwjV%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDczOTkzOTE3MzgxOSIMbI6OOCLFrPPfGqtfKrcDofGtGWmCsdGfX0QMBtnccCeH8nuddXGqwvRFTYjQL5wIKZ4M0WHJ%2B89fGTVB%2B69RiAKYzWJ1FU7U%2Bmdv%2BaMLxYDWeC1jnO48q2rqVtrtCXY0QkhUNGac0VTG8%2F9MK2FgbiNap4LPxDmFXy7%2FQSb1%2BnSPhsuUO%2B7EXU10bNqhF9Pq%2F2v8JJRqxjyORFTlflbisQOhvaTfFY%2BTPYR0Ej2fWdDmh47kw0mIKYf0c81eU%2FPJkcIwJZFg1FPVfzdY%2FQOiXQPWUfunjFoxZhll2iW6dVJ3ayyFmTACKa5s1Zj4mHdni9k2qW%2Fb9J9YFy1R7Ybfy6bLqj8FoONyhWskqA05dO7ZeyrbO%2F7DEMnuJPWxYGZ47z9TKoVmlC8fNHG%2BuVb1mAfg718GUzJzSCgZDesL0iwXtHN0vOsz28SkB3DehOO8o485613vZnlvjSFlbWTI%2FeXzqIqBazJn0dta6qlcsF6BdnB6gRyaYe3o0GxbjY7gIc4laVP5PTsz3Ixo9lsRuUwiaPMQsrSa%2BGByaxYQK1izpjB3whBynYrZ047%2FDta5fRjAaPqQrcrrxbRIm8EJt%2FoORboVyTDCxZDqBTq1AcD83IbPTZZuB4voJ5hM0VeuuBl32EYrauRg13nedNIVO%2FZhlDsuRw8K5P3%2BpFByJAWPY22VB1%2FJ4To2KYnVTuZJ1wgF5EFLQ4kSwz%2B7R%2BmECHKQadCqbphgR5oKw%2F6qJx8OA42wnSKvDh1nEccFzUrGkuCNdYNM%2BWWjySjiCXaO7WFsfDrnlpNXqTH2w4wyGjw73%2FY%2FGr9Ecdn4e%2B0ZO%2F3Zdhrbvz4WgWH%2B9xbc%2B0y2AzClO5Y%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190802T130540Z&X-Amz-SignedHeaders=host&X-Amz-Expires=599&X-Amz-Credential=ASIA2YR6PYW57Z6IZL7V%2F20190802%2Fus-west-1%2Fs3%2Faws4_request&X-Amz-Signature=bb35c80835947724a4977df5c9913d5574147da7479a53e0690cd806aebffdb8)



By seeing this PNG, I got horrified. I Hope I get good accuracy now. But still the accuracy is low.

Now its time to better go for Re-solution.
Either, I go back to resample and check out some more ML algorithms. Thinking about resampling is time consuming as i am a datascientist. I prefer to go with checking more classification algorithm. We have achieved 93% accuracy with my all time favorite random forest Model.
So this is how ML can beat DL.







