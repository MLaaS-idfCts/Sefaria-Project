def get_section(input_string):
    """
    Given a string, produce the substring that lies 
    after the last comma (if any) but 
    before the numbers at the end (if any).
    """
    result = input_string # init
    
    # get rid of everything before last comma
    last_comma = input_string.rfind(', ')
    if last_comma != -1:
        result = input_string[last_comma + 2:]

    # keep only letters and spaces
    result = ''.join(char for char in result if char.isalpha() or char == ' ')

    # remove single chars
    result = ' '.join( [w for w in result.split() if len(w)>1] )
    return result

input_list = [
        # "Gittin 78a:1",
        # "Shulchan Arukh, Even HaEzer 139:15",
        # "Mishneh Torah, Rest on a Holiday 6:12",
        # "Beitzah 17b:13",
        "Mishneh Torah, Sabbath 22:10"
    ]

for input in input_list: 
    output = get_section(input)
    print(output)