from transformers import pipeline, Conversation,BitsAndBytesConfig
import torch
import pandas as pd
import re
import nltk
import warnings
warnings.filterwarnings("ignore")

#nltk.download('punkt')
#nltk.download('all')
# load_in_8bit: lower precision but saves a lot of GPU memory
# device_map=auto: loads the model across multiple GPUs

#Load the file and make similar prompts

data = pd.read_csv(r"/data1/s3531643/thesis/Data/data4ektha.csv")
dataframe = data.groupby(["text.x"])["text.y"].count().reset_index()

# data["count of sentences"] = data["text.y"].apply(lambda x: len(nltk.sent_tokenize(str(x))))
# dataframe = data.groupby(["text.x"]).agg({"text.x": 'first',
#                                           'text.y' : lambda x: len(x),
#                      'count of sentences' : lambda x: list(x)})

chatbot = pipeline("conversational", model="BramVanroy/GEITje-7B-ultra",batch_size=4, model_kwargs={"load_in_4bit":True, "bnb_4bit_compute_dtype":torch.float16 },device_map="auto")
new_data_comments_generated = pd.DataFrame(columns = ["Post","Comments"])

total_rows = 0
total_matchs = 0

for i in range(len(dataframe)):

    print("I:",i,"POST: ",dataframe["text.x"][i], "Number of comments:",dataframe["text.y"][i])
   # print("I:",i,"POST: ",dataframe["text.x"][i], "Number of comments:",dataframe["text.y"][i],"Count of sentences",dataframe["count of sentences"][i])

    for j in range(dataframe["text.y"][i]):

        conversation = Conversation()
        # prompts = "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer een Facebook-reactie van een nieuwe gebruiker op het hierboven gespecificeerde bericht.".format(dataframe["text.x"][i])

        example_post = "'Schoonmaken. Opruimen. Foxhol winterklaar maken. Doe je mee?' Dat was het motto om de huurders in Foxhol aan te moedigen om in actie te komen. In Midden-Groningen houden we jaarlijks een actiedag waarin de leefbaarheid, in een wijk of dorp waar we veel woningen bezitten, centraal staat. Dit keer was Foxhol aan de beurt. Drie containers met een inhoud van veertig kuub lagen aan het eind van de middag vol met grof vuil! Tijdens de opruiming werden de broodjes hamburger liefdevol in ontvangst genomen door de bewoners, gepaard met koffie en thee. Al met al een succesvolle dag!"

        example_comments = [
            "Mag men ook wel eens in Musselkanaal doen....top actie… ",   #Positive
            "Zorg eerst  maar Dat .. West indischekade .. opgeruimd en netjes word .. dat zijn  in totaal 3 straten  3 flatten   Suc6 ..  ik steek me poten niet uit als jullie niet eens op of om kijken",   #Negative
            "Best wel jammer, oud jaar komt er weer aan. Je had er lekker warm bij kunnen zitten met al dat hout en matrassen " #Sarcasm?
        ]

        prompts = (
                "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst:\n"
                "{}\n\n"
                'Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht.'
                "Hier zijn enkele voorbeelden van een bericht en reacties op dat bericht:\n\n"
                "Voorbeeldbericht: {}\n"
                "Voorbeeldantwoord 1: {}\n"
                "Voorbeeldantwoord 2: {}\n"
                "Voorbeeldreactie 3: {}\n\n"
                "Nu jij! Genereer een nieuw antwoord op het bovenstaande bericht tussen dubbele aanhalingstekens."
            ).format(dataframe["text.x"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

        # prompts = "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer alleen {} Facebook-reacties van gebruikers op het hierboven gespecificeerde bericht. Genereer elke opmerking op een nieuwe regel.\n\nVoorbeeld:\nPost: \"{}\"\nReactie 1: \"{}\"\nReactie 2: \"{}\"\nReactie 3: \"{}\"".format(dataframe["text.x"][i], dataframe["text.y"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

        #print(dataframe["count of sentences"][i], max(dataframe["count of sentences"][i]),min(dataframe["count of sentences"][i]))

        # prompts = "Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer één Facebook-reactie van een gebruiker op het hierboven gespecificeerde bericht. De opmerking kan maximaal {} aantal zinnen bevatten. Genereer de opmerking tussen dubbele aanhalingstekens.".format(dataframe["text.x"][i],dataframe["count of sentences"][i])

        # prompts = """Op de Facebookpagina van een woningwebsite is het volgende bericht geplaatst: {}. Genereer één Facebook-reactie van een gebruiker op het hierboven gespecificeerde bericht. De opmerking kan maximaal {} aantal zinnen bevatten.  Genereer elke opmerking op een nieuwe regel.
        # Voorbeeld Post: \"{}\"
        # Reactie 1: \"{}\"
        # Reactie 2: \"{}\"
        # Reactie 3: \"{}\
        # Genereer de opmerking tussen dubbele aanhalingstekens.""".format(dataframe["text.x"][i], dataframe["count of sentences"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

        # prompts = "Het volgende bericht is op de Facebook-pagina van een woningwebsite geplaatst: {} Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht. De opmerking kan maximaal {} zinnen bevatten. Het volgende is een voorbeeld van een bericht en opmerkingen bij dat bericht. Voorbeeld Post: {} Voorbeeld Reactie 1: {} Voorbeeld Reactie 2: {} Voorbeeld Reactie 3: {}".format(dataframe["text.x"][i],dataframe["count of sentences"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

#         prompts = (
#     "Het volgende bericht is op de Facebook-pagina van een woningwebsite geplaatst:\n"
#     "{}\n\n"
#     "Genereer een Facebook-reactie van de gebruiker op het hierboven gespecificeerde bericht. "
#     "De opmerking kan maximaal {} zinnen bevatten.\n\n"
#     "Hier zijn enkele voorbeelden van een bericht en reacties bij dat bericht:\n\n"
#     "Voorbeeld Post: {}\n"
#     "Voorbeeld Reactie 1: {}\n"
#     "Voorbeeld Reactie 2: {}\n"
#     "Voorbeeld Reactie 3: {}\n\n"
#     "Nu jij! Genereer een nieuwe reactie op het bovenstaande bericht."
# ).format(dataframe["text.x"][i],dataframe["count of sentences"][i], example_post, example_comments[0], example_comments[1], example_comments[2])

        print(prompts)

        conversation.add_user_input(prompts)
        conversation = chatbot(conversation,
                               do_sample=True,
                               temperature=0.7,
        top_k=50,
        top_p=0.90)
        response = conversation.messages[-1]["content"]

        print("Response : ",response,"\n")

        #pattern = r'^(".+?"|```.+?```|".+?$)'  #Check for patterns or comments generated between double quotation marks "" . It may end abrubtly so it also checks for start of " but may not end with "

        pattern = r'(?:\"|\`\`\`)(.*?)(?:\"|\`\`\`|$)'

        # Find all matches
        matches = re.findall(pattern, response,re.DOTALL)

        # Print the matches
        for match in matches:
            if(matches!=[]):
                total_matchs+=1
                new_row = [dataframe["text.x"][i],  match]
                new_data_comments_generated.loc[len(new_data_comments_generated)] = new_row
                print("MATCH",match)
                print()

        if(matches==[]):
            new_row = [dataframe["text.x"][i],  response]
            new_data_comments_generated.loc[len(new_data_comments_generated)] = new_row


print("Total matches",total_matchs)
new_data_comments_generated.to_csv("Generated_comments_FewShot1060_Diverse990.csv")