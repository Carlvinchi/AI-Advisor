from joblib import load


# Function to Make Predictions for a single disease
def make_prediction_single(technique_name, input_matrix):
  try:
  # Load saved model
    path = "./model/"
    mod_name = technique_name
    saved_model = load(str(path + mod_name + ".joblib"))
    

  except Exception as e:
    print(e)
    print('\n')
    print("module Not Found")

  if input_matrix is not None:
    result = saved_model.predict(input_matrix)
    #accuracy = accuracy_score(test_labels, result)
    #clf_report = classification_report(test_labels, result)
    return result



param = ['What is your cumulative weighted average(CWA) / grade point average(GPA)?','How many times do you study every week?', 'Were you able to increase your CWA/GPA last semester?', 'Do you have a personal time table?', 'Do you always understand what you are taught at lectures?', 'How often have you been attending lectures?']


#preprocessing model to make single predictions
myInput = []
for i in range(0,11):
    myInput.append(0)
print(myInput)

input_vals = []

for i in range(0,len(param)):
    print(param[i])
    vals =input('Enter: ')
        #store input in a list
    input_vals.append(vals)

print(input_vals)

my_im_dict = {}
k = 0
for i in input_vals:
    my_im_dict[k]= i
    k+=1

print(my_im_dict)

    # preprocessing user input for the machine model
for i in range(0,len(myInput)):
    if i < 2:
        myInput[i] = my_im_dict[i]

    elif i == 2:
        if my_im_dict[i] == "No": myInput[i] = 1
        else: myInput[i+1] = 1
        

    elif i == 3:
        if my_im_dict[i] == "No": myInput[i+1] = 1
        else: myInput[i+2] = 1
        
        

    elif i == 4:
        if my_im_dict[i] == "No": myInput[i+2] = 1
        else: myInput[i+3] = 1
        
        

    elif i == 5:
        if my_im_dict[i] == "seldom": myInput[i+3] = 1
        elif my_im_dict[i] == "often" : myInput[i+4] = 1
        else: myInput[i+5] = 1

print(myInput)

input_mat = [myInput]

model_name = input('Enter model name: ')
    #Testing for a single disease

res = make_prediction_single(model_name,input_mat)
print('Prediction by:', model_name)
print(res[0])



