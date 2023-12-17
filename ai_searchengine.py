from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import easyocr
import os
import pandas
import tkinter
from tkinter import StringVar, filedialog, messagebox, simpledialog, ttk, Button
import threading
from sys import exit
import torch
import pickle

picdata=[]
sentences=[]
datafile=[]
filename=''

#Sets up OCR Scanner and ML model, I think GPU mode only works with NVIDIA Gpu's unfortunately
currPath=os.getcwd()
gpu=torch.cuda.is_available()
if gpu==True:
    ocr_model= easyocr.Reader(['en'], gpu=True, model_storage_directory=currPath+"\\"+'easyocr',download_enabled=False)
else:
    ocr_model= easyocr.Reader(['en'], gpu=False, model_storage_directory=currPath+"\\"+'easyocr' ,download_enabled=False)

model = SentenceTransformer(currPath+"\\"+'MiniLM')

#Iterates through directories of given source and gets list of file paths of image type
def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')):
                matches.append(os.path.join(root, filename))
    return matches

#Indexes given folder: Goes through the list of file paths using fileList then scans the images with the OCR creating a pickle file and storing it in the directory then goes through the pickle file and creates embeddings (vectors) of the extracted texts and again stores it in the directory
def index():

    #Gets path from user input
    inputpath=folder_path.get()
    if inputpath != '':
        os.chdir(filename)
    else:
        messagebox.showwarning("Error", "Please browse to the folder with the images before indexing.")
        startIndex.config(state='normal')
        return
    
    #Clears any previous results in secondframe so new ones can be displayed, this is important if the user searched for something else or was in another directory previously.
    for widget in secondframe.winfo_children():
        widget.destroy()
    
    #Calls fileList method to get all image paths from directory and subdirectories
    matches=fileList(inputpath)
    
    #Scans every image with OCR, capturing the text
    length_of_matches=len(matches)
    tkinter.Label(secondframe, font=("Helvetica", 9), bg="#EDEADE", width=41, anchor='n', text="Scanning images, progress at.. 0%").grid(row=0, column=0, padx=10, pady=10 )
    for file_path in matches:
        IMAGE_PATH= file_path
        try:
            result = ocr_model.readtext(IMAGE_PATH)

        # Skips empty images
        except AttributeError:
            # print('Image was none')  # take comment out to see how many images are skipped
            progress['value'] += (100/length_of_matches)
            continue
        
        # Creates a green progress bar at the very bottom of the GUI, every time the loop is run the progress is increased.
        progress['value'] += (100/length_of_matches)
        progress_amount=str(int(progress['value']))
        tkinter.Label(secondframe, font=("Helvetica", 9), bg="#EDEADE", width=41, anchor='n', text="Scanning images, progress at.. "+progress_amount+"%").grid(row=0, column=0, padx=10, pady=10 )

        # Concatenates scanned sentences together. (This needs optimization for long strings, essentially some basic form of tokenization, 
        # for ex: if len(content)>150, create array with content[1] at first fullstop after 150, etc. otherwise the embeddings lose meaning in big texts)
        content=""
        for sentence in result:
            content= str(content)+' '+str(sentence[1])
        # print(content) # take comment out to see how well the OCR is reading the images

        # Adds each image content to a list/database
        contentArray = (IMAGE_PATH, content)
        picdata.append(contentArray)
        
    # Converts database into datafile to be saved and stored in local directory
    datafile = pandas.DataFrame(picdata)
    with open('index.pkl','wb') as f:
            pickle.dump(datafile, f)
    
    # Embeds sentences in datafile, this will create a set of NLP embeddings that can be used to do ai searches with sentence similarity.
    for nonstandardized_sentence in datafile[1]:
        standardized_sentence=nonstandardized_sentence.lower()
        sentences.append(standardized_sentence)

    global embeddings
    embeddings = model.encode(sentences)

    #Saves embeddings to local directory
    torch.save(embeddings, "AIE.pt") 
    progress['value'] =0

    #Success message to let user know indexing is done and he can now search
    messagebox.showinfo ("Success!", "Pictures are indexed.")
    startIndex.config(state='normal')


#Performs AI search using NLP sentence similarity
def AIsearch(query):

    #Gets user query
    if query != '':
        pass
    else:
        messagebox.showwarning("Error", "Please input your query in the search box.")
        startSearch.config(state='normal')
        return
    
    #Gets path from user input
    inputpath=folder_path.get()
    if inputpath != '':
        os.chdir(filename)
    else:
        messagebox.showwarning("Error", "Please browse to the folder with the indexed images before searching.")
        startSearch.config(state='normal')
        return

    # Embeds the query
    standard_query = query.lower()
    queryArray = [standard_query]
    inputEmb = model.encode(queryArray)

    # Loads embeddings and compares them to query, this is the bulk of ai search, it's a similarity function.
    try:
        embeddings= torch.load("AIE.pt")
    except FileNotFoundError:
        messagebox.showwarning("Error", "You didn't index the pictures, index then try again")
        startSearch.config(state='normal')
        return
    try:
        similarity=cosine_similarity(
            [inputEmb[0]],
            embeddings[0:]
        )
    except ValueError:
        messagebox.showwarning("Error", "You didn't index the pictures, index then try again")
        startSearch.config(state='normal')
        return
    # These is the similarity values list, in order of index not of highest score.
    similarityList=(similarity[0])

    #Load indexed pictures to get their paths and associate with similarity list.
    try:
        with open('index.pkl','rb') as f:
                    datafile = pickle.load(f)
    except FileNotFoundError:
        messagebox.showwarning("Error", "You have yet to index this directory, please index then try searching again")
        normSearch.config(state='normal')
        return
    # This is the indexed pictures paths list.
    dirList=list(datafile[0])

    # Zips results together so that paths and embeddings order can be manipulated this part would need to be modified if tokenization was implemented
    nested_results= [[path, value] for path, value in zip(dirList, similarityList)]
    # Change to descending value order, so it displays the most relevant result at the top
    sorted_results= sorted(nested_results, key=lambda x: x[1], reverse=True)
    
    #Returns and display results
    index = 0
    if sorted_results != []:

        # For loop for tkinter, made to display all results as interactive paths that can be opened. So the user can simply open the relevant result.
        for x,y in sorted_results:
            lb=tkinter.Label(secondframe, font=("Helvetica", 9), bg="#EDEADE", width=41, anchor='e', text=x+'_'+str(y)) #path prints out
            lb.grid(row=index, column=0, padx=10, pady=10 )
            Button(secondframe, text="open",  borderwidth=1, command = lambda x=x: os.startfile(x, 'open')).grid(row=index, column=1, pady=10) #button to open path
            index += 1

    else:
        messagebox.showinfo("No results", "Sorry, no results were found.")

    # Resets button so it's clickable again, returns and display results
    startSearch.config(state='normal')
    return    


# Performs Keyword search
def normSearchMethod(query):
    
    # Gets query from user input
    if query != '':
        pass
    else:
        messagebox.showwarning("Error", "Please input your query in the search box.")
        normSearch.config(state='normal')
        return
    
    # Gets path from user input
    inputpath=folder_path.get()
    if inputpath != '':
        os.chdir(filename)
    else:
        messagebox.showwarning("Error", "Please browse to the folder with the images before indexing.")
        startIndex.config(state='normal')
        return

    # Checks for indexed pkl file in folder and loads indexed pictures
    if os.path.isfile('index.pkl') ==True:
        with open('index.pkl','rb') as f:
                    datafile = pickle.load(f)
    else:
        messagebox.showwarning("Error", "You have yet to index this directory, please index then try searching again.")
        normSearch.config(state='normal')
        return
    
    # Clears any previous results in secondframe so new ones can be displayed
    for widget in secondframe.winfo_children():
        widget.destroy()

    # Normalize list so it's case insensitive
    dirList=list(datafile[1])
    listNorm=[]
    for word in dirList:
        listNorm.append(word.lower())

    # Finds all results in list (and normalizes query)
    final_result=[]
    index_count=0
    length_list=len(listNorm)
    while index_count<length_list:
        if query.lower() in listNorm[index_count]:
            final_result.append(datafile[0][index_count])
            index_count += 1
            pass
        else:
            index_count += 1
    
    # Returns and display results
    i = 0
    if final_result != []:

        # For loop for tkinter, made to display all results as interactive paths that can be opened. So the user can simply open the relevant result.
        for x in final_result:
            lb=tkinter.Label(secondframe, font=("Helvetica", 9), bg="#EDEADE", width=41, anchor='e', text=x) #path prints out
            lb.grid(row=i, column=0, padx=10, pady=10 )
            Button(secondframe, text="open",  borderwidth=0, command = lambda x=x: os.startfile(x, 'open')).grid(row=i, column=1, pady=10) #button to open path
            i += 1

    else:
        messagebox.showinfo("No results", "Sorry, no results were found.")
    normSearch.config(state='normal')
    return


# Converts Indexed file to csv to export image texts
def savetoCSV():

    # Checks if directory has been selected
    inputpath=folder_path.get()
    if inputpath != '':
        os.chdir(filename)
    else:
        messagebox.showwarning("Error.", "Please browse to a folder with indexed images before searching.")
        return

    # Checks if there is indexed file in directory
    if os.path.isfile('index.pkl') ==True:
        with open('index.pkl','rb') as f:
                    datafile = pickle.load(f)
        datafile.to_csv('IndexedImages.csv')
        messagebox.showinfo ("Success!", "IndexedImages.csv has been created in the selected folder.")
    else:
        messagebox.showwarning("Error.", "You didn't index the pictures or have navigated to the wrong folder, index then try again")
        saveCSV.config(state='normal')
        return


# BUTTON FUNCTIONS
# Browse button function
def browse_button():
    global folder_path
    global filename
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    os.chdir(filename)

# AI Search button function
def startSearchThread():
    query=queryEntry.get()
    startSearch.config(state='disabled')
    threading.Thread(target=AIsearch, args=(query,), daemon=True).start()

# NormalSearch button function
def startNormSearch():
    query=queryEntry.get()
    normSearch.config(state='disabled')
    threading.Thread(target=normSearchMethod, args=(query,), daemon=True).start()

# Index button function
def startIndexThread():
    startIndex.config(state='disabled')
    threading.Thread(target=index, args=(), daemon=True).start()

# Quit button function
def endThread():
    exit()

# Lets the frame be scrollable with mouse or gesture when focused on (ie. hovered in or clicked)
def on_vertical(event):
    my_text.yview_scroll(-1 * int(event.delta/120), 'units')

# Lets the app press the search button when the user presses Enter
def on_enter(event):
    startNormSearch()

# GUI
root = tkinter.Tk()
root.title("Search Engine for Images")
root.resizable(width=False, height=False)
root.geometry("450x650")
root.configure(background='#181A1C')

frame= tkinter.Frame(root, bg="#181A1C")
frame.place(relx=.1, rely=.05, relwidth=.8, relheight=.9)

# Progressbar at the bottom
progress = ttk.Progressbar(root, orient = 'horizontal', length = 100, mode = 'determinate')
progress.place(relx=0, rely=.99, relwidth=1, relheight=.1)

# TOP SECTION
# Query Label 
queryL = tkinter.Label(frame, text="Query: ", fg="white", bg="#181A1C") #, padx=1, pady=1,
queryL.place(relx=.05, rely=.01, relwidth=.1, relheight=.05)

# Query Input box
queryEntry=tkinter.Entry(frame, bg='#c5c7c4')
queryEntry.place(relx=.2, rely=.01, relwidth=.75, relheight=.05)

# Index Label 
folder_path = StringVar()
indexL = tkinter.Label(frame, text="Index: ", fg="white", bg="#181A1C") #, padx=1, pady=1,
indexL.place(relx=.05, rely=.1, relwidth=.1, relheight=.05)

# Index Label for path
indexP = tkinter.Label(frame, textvariable=folder_path, padx=0.5, pady=5, fg="white", bg="#26292c", anchor='e')
indexP.place(relx=.2, rely=.1, relwidth=.5, relheight=.05)

# Index Button (Browse)
startBrowse=tkinter.Button(frame, text="Browse", padx=1, pady=1, activebackground="#181A1C", bg="#52595d", command= browse_button) 
startBrowse.place(relx=.75, rely=.1, relwidth=.2, relheight=.05)


# MID SECTION
# Text box for displaying and scrollbar
my_text = tkinter.Canvas(frame)
my_text.place(relx=0, rely=.2, relwidth=1, relheight=.625) # 360 x 365.625

my_scrollbar = tkinter.Scrollbar(frame, orient='vertical', command=my_text.yview)
my_scrollbar.place(relx=.96, rely=.2, relwidth=.04, relheight=.625)

my_text.configure(yscrollcommand=my_scrollbar.set)
my_text.bind('<Configure>', lambda e: my_text.configure(scrollregion=my_text.bbox("all")))

secondframe = tkinter.Frame(my_text)
my_text.create_window((0,0), window=secondframe, anchor="nw")

secondframe.bind('<Configure>', lambda e: my_text.configure(scrollregion=my_text.bbox("all")))  #refreshes the scrollbar!!!

# Lets the frame be scrollable with mouse or gesture when focused on (ie. hovered in or clicked)
root.bind_all('<MouseWheel>', on_vertical)
# Lets the app press the search button when the user presses Enter
root.bind('<Return>', on_enter)

# BOTTOM SECTION
# Buttons bot
startSearch=tkinter.Button(frame, text="AI Search", padx=1, pady=1, fg="white", bg="#52595d", command= startSearchThread) 
startSearch.place(relx=.025, rely=.875, relwidth=.3, relheight=.05)

normSearch=tkinter.Button(frame, text="Search", padx=1, pady=1, fg="white", bg="#52595d", command= startNormSearch)
normSearch.place(relx=.675, rely=.875, relwidth=.3, relheight=.05)

saveCSV=tkinter.Button(frame, text="Save to CSV", padx=1, pady=1, fg="white", bg="#52595d", command= savetoCSV)
saveCSV.place(relx=.35, rely=.875, relwidth=.3, relheight=.05)

startIndex=tkinter.Button(frame, text="Index", padx=1, pady=1, fg="white", bg="#52595d", command= startIndexThread)
startIndex.place(relx=.15, rely=.95, relwidth=.3, relheight=.05)

endScript=tkinter.Button(frame, text="Quit", padx=1, pady=1, fg="white", bg="#52595d", command= endThread)
endScript.place(relx=.55, rely=.95, relwidth=.3, relheight=.05)

root.mainloop()