from typing import Dict

def get_aliases(module_config: Dict) -> Dict:
  """Returns a dict containing a mapping of `alias:model`"""
  aliases = {}
  for model, properties in module_config.items():
    if type(properties) != dict: continue
    if "alias" not in properties.keys(): continue
    
    aliases[properties["alias"]] = model

  return aliases

def validate_brackets(equation):
    """
    Validates if brackets in the compressed layer configuration are balanced.
    
    Checks for:
    - Every opening bracket has a matching closing bracket
    - Brackets are properly nested
    - Supports (), [], and {} bracket types
    
    Args:
        layer_config (str): A string containing the compressed layer configuration
        
    Returns:
        bool: True if brackets are correctly balanced, False otherwise
    """
    # Dictionary mapping opening brackets to their closing counterparts
    brackets_map = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    
    # Stack to keep track of opening brackets
    stack = []
    
    for char in equation:
        # If it's an opening bracket, push it onto the stack
        if char in brackets_map:
            stack.append(char)
        
        # If it's a closing bracket
        elif char in brackets_map.values():
            # If stack is empty, there's no matching opening bracket
            if not stack:
                return False
            
            # Get the last opening bracket
            last_opening = stack.pop()
            
            # Check if the current closing bracket matches the expected one
            if brackets_map[last_opening] != char:
                return False
    
    # After processing all characters, stack should be empty
    # If not empty, there are unmatched opening brackets
    return len(stack) == 0

def layer_parser(layer_config: str, aliases: Dict):
  """
  Expands an compressed layer configuration string into an expanded list of layers
  
  For example:
  "Attn, 2(M2, 2(M1, SWA))" -> ["Attn", "M2", "M1", "SWA", "M1", "SWA", "M2", "M1", "SWA", "M1", "SWA", "Attn"]
  
  Args:
    layer_config (str): The input string containing the layer configuration
      
  Returns:
    list: The expanded list of operations
  """
  # Remove any whitespace
  layer_config = layer_config.replace(" ", "")
  assert validate_brackets(layer_config), "Brackets are not balanced"
  
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
  expanded_list, _ = parse_expression(layer_config)

  final_list = []

  # Convert from aliases to standardised model names
  for layer_name in expanded_list:
      if layer_name in aliases.keys():
          final_list.append(aliases[layer_name])
      else:
          final_list.append(layer_name)

  return expanded_list

aliases = get_aliases({
    "mamba2": {
       "alias": "M2"
    },
    "mamba1": {
      "dummy": 1,
       "alias": "M1"
    },
    "multiheadattention": {
       "alias": "MHA",
       "dummy": 1
    },
    "attention": {
       "alias": "Attn"
    },
    "test comment": 123
})
lst = layer_parser("Attn, 3 (M2, 2(M1, MHA)), Attn, Unaliased", aliases)
print(lst)
