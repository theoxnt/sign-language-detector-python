def ask(prompt, cast_type=None, min=None, max=None):
    """
    Ask the user for input with optional type casting and range validation.

    Args:
        prompt (str): The prompt message to display to the user.
        cast_type (type, optional): The type to cast the input to (e.g., int, float, str). Defaults to None.
        min (optional): Minimum acceptable value (inclusive). Defaults to None.
        max (optional): Maximum acceptable value (inclusive). Defaults to None.

    Returns:
        The user's input.
    """
    if not type(prompt) is str:
        raise TypeError("prompt must be a string")
    if prompt == "":
        raise ValueError("prompt must be not empty")
    if cast_type != None and min != None and cast_type != type(min):
        raise TypeError("min must be the same type as cast_type")
    if cast_type != None and max != None and cast_type != type(max):
        raise TypeError("max must be the same type as cast_type")
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
    """
    Print a prompt message to the user.
    """
    print(prompt)
