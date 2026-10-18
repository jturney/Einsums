# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.



__block_starts = ["block", "foreach", "function", "if", "macro", "while"]
__block_mids = ["else", "elseif"]
__block_ends = ["endblock", "endforeach", "endfunction", "endif", "endmacro", "endwhile"]

def cleanup_cmake(file_string, indent = 2) :
    block_depth = 0
    paren_depth = 0
    arg_depth = 0

    empty_lines = 0

    out = ""

    for line in file_string.split("\n") :
        stripped = line.strip()
        # Check for empty lines
        if len(stripped) == 0:
            if empty_lines == 0 :
                out += "\n"
            empty_lines += 1
            continue
        empty_lines = 0

        # Check to see if the line begins with block-middle or block-end identifier. These identifiers don't indent.
        first_ident = ""
        for ch in line :
            if ch.isalpha() :
                first_ident += ch
            else :
                break

        depth = block_depth + paren_depth
        if first_ident in __block_mids :
            depth -= 1
        elif first_ident in __block_ends :
            depth -= 1
            block_depth -= 1
        elif first_ident in __block_starts :
            block_depth += 1
            # Don't set depth here yet.

        # Check to see if the first identifier is a capitalized argument or not.
        if paren_depth > 0 and first_ident.isupper() :
            arg_depth = 1
        elif paren_depth > 0 and arg_depth == 1 :
            depth += 1


        # Parse the rest of the line for unclosed parentheses.
        paren_before = paren_depth
        for ch in stripped :
            if ch == '#' :
                break
            if ch == '(' :
                paren_depth += 1
            elif ch == ')' :
                paren_depth -= 1
        if paren_before > paren_depth :
            arg_depth = 0

        # Indent.
        out += " " * depth * indent

        # Output the string, removing extra spaces.
        spaces = 0
        quoted = False
        comment = False
        for ch in stripped :
            if comment :
                out += ch
            elif ch.isspace() :
                if spaces == 0 or quoted :
                    out += ch
                spaces += 1
            elif ch == '"':
                spaces = 0
                quoted = not quoted
                out += ch
            elif ch == '#' :
                spaces = 0
                out += ch
                comment = True
            else :
                spaces = 0
                out += ch
        out += '\n'

    # Make sure the file ends in a new line.
    if len(out) == 0 or out[-1] != '\n' :
        out += '\n'
    return out
