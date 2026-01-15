

def ask(prompt, cast_type=None, min=None, max=None):
    valide = False
    while not valide:
        response = input(prompt)
        if cast_type:
            try:
                response_typed = cast_type(response)
            except ValueError:
                print(f"Please enter a valid {cast_type.__name__}.")
                continue
        if min is not None and response_typed < min:
            print(f"Please enter a value greater than or equal to {min}.")
            continue
        if max is not None and response_typed > max:
            print(f"Please enter a value less than or equal to {max}.")
            continue
        valide = True
    return response

def print_prompt(prompt):
    print(prompt)