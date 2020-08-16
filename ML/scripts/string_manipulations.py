with open("images/scores/scores_key.txt", "w") as file_object:
    file_object.write("blah")
with open("images/scores/scores_key.txt", "a") as file_object:
    file_object.write("hello")