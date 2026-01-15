import os
from core import collect_images, create_dataset, train_classifier, inference_classifier
from io_ import ask, print_prompt
    
def main():
    response = ask("1- Collect images\n"
    "2- Create dataset\n"
    "3- Train model\n"
    "4- Use model\n"
    "5- Quit\n"
    "\n", cast_type=int, min=1, max=5)

    action_finished = False
    if response == "1":
        imgs_per_class = ask("\nHow many images do you want to collect per class? ", cast_type=int, min=1, max=1000)
        num_classes = ask('How many classes do you want to collect images for? ', cast_type=int, min=1, max=26)
        data_folder_name = ask("Enter the name of the folder to save the images (will be created if it doesn't exist): ", cast_type=str)
        action_finished = collect_images(int(num_classes), int(imgs_per_class), data_folder_name)
        if not action_finished:
            print_prompt("Image collection failed.")
            return

    elif response == "2":
        number_of_classes = ask("\nFor how many class do you want to create a dataset? ", cast_type=int, min=1, max=26)
        dataset_name = ask("Enter the name for the dataset file (without extension): ", cast_type=str)
        action_finished = create_dataset(int(number_of_classes), dataset_name)
        if not action_finished:
            print_prompt("Dataset creation failed.")
            return
            
        
    elif response == "3":
        data_file = ask("\nEnter the dataset folder name (without extension) to use for training: ", cast_type=str)
        DATA_DIR = '/home/theoxnt/pologne/PITE/sign-language-detector-python/src/data_pickle'
        if not os.path.exists(os.path.join(DATA_DIR, f'{data_file}.pickle')):
            print_prompt("Dataset file does not exist. Please create the dataset first.")
            return
        else:
            type_valide = False
            while not type_valide:
                training = ask("Do you want to train with random forest or with machine learning? (Enter 'f' or 'ml'): ", cast_type=str) 
                if training == 'f':
                    type_valide = True
                    print_prompt("Training with Random Forest...")
                    action_finished = train_classifier(data_file, training) 
                elif training == 'ml':
                    type_valide = True
                    print_prompt("Training with Machine Learning...")
                    num_classes = ask("For how many classes was the dataset created? ", cast_type=int, min=1, max=26)
                    action_finished = train_classifier(data_file, training, int(num_classes))
                else:
                    print("Invalid option selected. Please enter 'f' or 'ml'.")
            if not action_finished:
                print_prompt("Model training failed.")
                return
            
    elif response == "4":
        type_valide = False
        while not type_valide:
            model_type = ask("\nWhich model do you want to use? (Enter 'f' for random forest or 'ml' for machine learning): ", cast_type=str) # fonction à mettre dans le core
            if model_type == 'f':
                type_valide = True
            elif model_type == 'ml':
                type_valide = True
            else:
                print("Invalid option selected. Please enter 'f' or 'ml'.")
        print_prompt("Using model...")
        action_finished = inference_classifier(model_type) # fonction à mettre dans le core
        if not action_finished:
            print_prompt("Model inference failed.")
            return


    elif response == "5":
        print_prompt("Quitting...")
        return
    else:
        raise ValueError("Option not recognized.")   
    if action_finished:
        print_prompt("\nAction completed successfully. What do you want to do next?\n")
        return main()
    
if __name__ == "__main__":
    print_prompt("\nWelcome to the Sign Language Detector!\n" \
        "What do you want to do?\n")
    main()