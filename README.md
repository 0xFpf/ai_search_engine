# ai_search_engine
A local search engine that is useful to extract text from images and search them or save them locally into a csv.

This was a project that I needed for myself due to the high amount of screenshots I have in storage but that I can't export or search through.

It also makes use of NLP AI so that you don't need the exact keywords to search for an image. If the image has keywords such as 'cardiovascular fitness' and you input 'heart strength' it will find that picture.

I have made a (private) MacOS version that leverages Apple's ML Vision Core technology and also allows for all text files such as PDFs and Docs to be read and searched. 

This is in a way a (slower) competitor to the native Finder app, with the added benefit that it can scan images and do AI search.

There is plenty of room for optimization such as implementing tokenization and improving speed but have yet to decide whether or not to pursue it.

Another optimization could be made in automatically updating the data files when images are added or subtracted to a directory. Currently you have to re-scan the directory when images are added/subtracted.

Old demo here (w/out AI Search):



https://github.com/0xFpf/ai_search_engine/assets/74162889/ef59b84b-a96e-4c45-a8e9-d4be1a2a1196

