import glob2 ##Package helps to read multiple files
import pandas as pd #package helps to read the files and do various operations on the data.

#####################text to csv ########################
def text_to_csv(): #This method helps to convert the all text files into csv files.
    con_filenames = glob2.glob('D:/MA/MA_dataset/Preproccessed/Dementia bank/dementia/*.txt')  # list of all .txt files in the directory
    for i in con_filenames:
        df = pd.read_fwf(i) #reading the file
        p=i.split('D:/MA/MA_dataset/Preproccessed/Dementia bank/dementia')[1] #Here splitting the name of the text files so that csv files will save it by their original names.
        p=p.split('.txt')[0]
        df.to_csv('D:/MA/MA_dataset/Preproccessed/d1'+p+'.csv',index=None,header=False) #Here saving the dataframe to csv files.

    #########################merging rows into one row ########################
def merge_row_in_one_row(): #After getting the csv files this method helps to merge every row into single row
    con_filenames = glob2.glob('E:/MA/MA_dataset/Preproccessed/dementia_csv/*.csv') #list of all .csv file in the directory.
    for i in con_filenames: #iterating each csv file from the dementia_csv directory
        p=i.split("D:/MA/MA_dataset/Preproccessed/dementia_csv")[1] #Here splitting the name of the text files so that csv files will save it by their original names.
        fIn = open(i, "r",encoding="utf8")
        fOut = open("D:/MA/MA_dataset/Preproccessed/Dementia bank/dementia_csv_merge_data"+p, "w",encoding="utf8")
        fOut.write(",".join([line for line in fIn]).replace("/n","")) #using next line as a delimiter to join the rows.
        fIn.close() #closing the opened file.
        fOut.close() #closing the mereged row file.

###########merging columns into one#################
def merge_col_one_col(): #This method helps to merge the multiple columns into one.
    con_filenames = glob2.glob('E:/MA/MA_dataset/Preproccessed/Dementia bank/dementia_csv_merge_data/*.csv') #list of all .csv file after merging into single row and column in the directory.
    for i in con_filenames:
        p=i.split("E:/MA/MA_dataset/Preproccessed/Dementia bank/dementia_csv_merge_data")[1] #splitting the name of the file with the directory path.
        df=pd.read_csv(i,header=None) #reading the csv file.
        df['message'] = df[df.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1) #labelling the new column as message.
        df.drop(df.columns.difference(['message']), 1, inplace=True) #droping the rest of the columns and including the one new column named as message to keep the merge data.
        # df.insert ( 0 , "Label" , ["control"] , True )
        df.to_csv ('E:/MA/MA_dataset/Preproccessed/Dementia bank/dementia_csv_merge_data_columns'+ p , index=None,header=None ) #writing dataframe to csv file.

##########making one file of multiple files #####################
def making_one_file_of_multiple_file(): #this method helps to make one file of multiple files.
    con_filenames = glob2.glob('E:/MA/MA_dataset/Preproccessed/Dementia bank/dementia_csv_merge_data_columns/*.csv')  # list of all .txt files in the directory
    with open('dementia_output_files@.csv','wb') as newf: #creating dementia_output_file@ to make this a single file out of multiple files.
        for filename in con_filenames: #iterating through multiple files.
            with open(filename,'rb') as hf:
                newf.write(hf.read())
                #newf.write (b'/n')

##########adding label column and naming second column as message ################
def add_label_and_naming_rest_col(): #this method is adding label as dementia corresponding to the dementia message.
    df=pd.read_csv('D:/MA/dementia_output_files@.csv',header=None) #reading the file
    df.columns=['message'] #creating new column
    df.insert ( 0 , "Label" , "dementia" , True )
    df.to_csv('dementia_output_files@.csv',index=None)
    print(df.columns)

def label_count(): #this method is optional to count the labels over here.
    df=pd.read_csv('E:/MA/final_output_file@.csv')
    print(df['Label'].value_counts())

####Second Time without merging rows making dementia output file #######
def second_time_merge_rows_file(): #This method is the another way to preprocesss the data. Here we are making dementia output file without merging rows.
    con_filenames = glob2.glob('E:\MA\MA_dataset\Preproccessed\dementia_csv/*.csv')  # list of all .txt files in the directory
    with open('dementia_output_files@@.csv','wb') as newf:#creating dementia_output_file@@ to make this a single file out of multiple files withpout merging rows.
        for filename in con_filenames:
            with open(filename,'rb') as hf: #opening the file.
                newf.write(hf.read())
                # newf.write (b'/n')

######Same thing with control output file ##################
def second_time_control_file(): #This method is doing the same thing on the control files.
    import glob2
    con_filenames = glob2.glob('E:\MA\MA_dataset\Preproccessed\Dementia bank\control_csv/*.csv')  # list of all .txt files in the directory
    with open('control_output_files@@.csv','wb') as newf:
        for filename in con_filenames:
            with open(filename,'rb') as hf:
                newf.write(hf.read())

#############adding label column and naming second column s message ################
def add_label_second_time():
    df=pd.read_csv('E:\MA\dementia_output_files@@.csv',header=None, error_bad_lines=False) #read the csv file.
    df.columns=['message']
    df.insert ( 0 , "Label" , "dementia" , True )
    df.to_csv('dementia_output_files@@.csv',index=None)#writing dataframe to csv file.

############adding 5 column into one of control ########################
def add_five_col_one_control(): #this methods is merging th columns of control files into one column.
    df=pd.read_csv('E:\MA\control_output_files@@.csv',header=None, error_bad_lines=False)
    df['message'] = df[df.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    df.drop(df.columns.difference(['message']), 1, inplace=True)
    df.insert ( 0 , "Label" , "control" , True )
    df.to_csv ('control_output_files@@.csv',index=None)#writing dataframe to csv file.

if __name__ == '__main__': #to call all the methods in a sequential manner.
    text_to_csv()
    merge_row_in_one_row()
    merge_col_one_col()
    making_one_file_of_multiple_file()
    add_label_and_naming_rest_col()
    label_count()
    second_time_merge_rows_file()
    second_time_control_file()
    add_label_second_time()
    add_five_col_one_control()
    add_five_col_one_control()



