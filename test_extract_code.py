import ast
from extract.code import extract_classes_and_functions_from_ast
from core.logger import LoggerSetup

# Enable console logging for debugging
logger = LoggerSetup.get_logger("extract.code", console_logging=True)

# Sample Python code to parse
sample_code = '''
class SampleClass:
    def method_one(self):
        pass

def standalone_function():
    pass
'''

# Parse the sample code into an AST
tree = ast.parse(sample_code)

# Extract classes and functions
result = extract_classes_and_functions_from_ast(tree, sample_code)

# Print the result to verify the 'name' property is present
print("Classes extracted:")
for cls in result['classes']:
    print(f"Class name: {cls.get('name', 'No name')}")  # Use get method to safely access 'name'

print("\nFunctions extracted:")
for func in result['functions']:
    print(f"Function name: {func.get('name', 'No name')}")  # Use get method to safely access 'name'
