"""
Script to resolve merge conflicts by keeping HEAD version
"""

def resolve_conflicts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by conflict markers
    parts = []
    current_pos = 0
    
    while True:
        # Find next conflict start
        conflict_start = content.find('<<<<<<< HEAD', current_pos)
        if conflict_start == -1:
            # No more conflicts, add remaining content
            parts.append(content[current_pos:])
            break
        
        # Add content before conflict
        parts.append(content[current_pos:conflict_start])
        
        # Find conflict middle and end
        conflict_middle = content.find('=======', conflict_start)
        conflict_end = content.find('>>>>>>>', conflict_middle)
        
        if conflict_middle == -1 or conflict_end == -1:
            print(f"Warning: Malformed conflict at position {conflict_start}")
            break
        
        # Extract HEAD version (our changes)
        head_version = content[conflict_start + len('<<<<<<< HEAD\n'):conflict_middle]
        
        # Add HEAD version (keeping our changes)
        parts.append(head_version)
        
        # Move past the conflict
        end_of_line = content.find('\n', conflict_end)
        current_pos = end_of_line + 1 if end_of_line != -1 else conflict_end + 20
    
    # Write resolved content
    resolved = ''.join(parts)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(resolved)
    
    print(f"✓ Resolved conflicts in {filepath}")

if __name__ == "__main__":
    resolve_conflicts("example agents/agents.py")
    print("All conflicts resolved!")
