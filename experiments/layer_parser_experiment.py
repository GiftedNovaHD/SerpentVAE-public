def expand_attention_notation(input_str):
    """
    Expands an attention notation string into a list of operations.
    
    For example:
    "Attn, 2(M2, 2(M1, SWA))" -> ["Attn", "M2", "M1", "SWA", "M1", "SWA", "M2", "M1", "SWA", "M1", "SWA", "Attn"]
    
    Args:
        input_str (str): The input string containing attention notation
        
    Returns:
        list: The expanded list of operations
    """
    # Remove any whitespace
    input_str = input_str.replace(" ", "")
    
    def parse_expression(expr, idx=0):
        """Parse an expression starting from index idx"""
        result = []
        current_token = ""
        
        while idx < len(expr):
            char = expr[idx]
            
            if char.isalnum() or char == ',':
                if char == ',':
                    if current_token:
                        result.append(current_token)
                        current_token = ""
                else:
                    current_token += char
                idx += 1
            elif char == '(':
                if current_token:
                    # Handle repetition
                    repetitions = 1
                    if current_token[0].isdigit():
                        repetitions = int(current_token[0])
                        current_token = current_token[1:]
                    
                    # Parse the subexpression
                    subexpr, new_idx = parse_expression(expr, idx + 1)
                    
                    # Add the subexpression repeated times
                    for _ in range(repetitions):
                        result.extend(subexpr)
                    
                    current_token = ""
                    idx = new_idx
                else:
                    # If no token before '(', just parse the subexpression
                    subexpr, new_idx = parse_expression(expr, idx + 1)
                    result.extend(subexpr)
                    idx = new_idx
            elif char == ')':
                if current_token:
                    result.append(current_token)
                return result, idx + 1
            
        if current_token:
            result.append(current_token)
        
        return result, idx
    
    # Parse the input string
    expanded_list, _ = parse_expression(input_str)
    return expanded_list

input_str = "Attn, 2(M2, 3(M1, SWA)), Attn"
result = expand_attention_notation(input_str)
print(result)