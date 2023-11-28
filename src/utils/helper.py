import string

def sanitize_file_path(file_path):
    valid_chars = string.ascii_letters + string.digits + "/._ "
    sanitized_path = ''.join(c for c in file_path if c in valid_chars)
    sanitized_path = sanitized_path.replace(' ', '_')
    return sanitized_path