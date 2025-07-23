import numpy as np
from tabulate import tabulate

def get_matrix(name):
    while True:
        size = input(f"Enter the size for {name} ").lower().replace('*', 'x')
        try:
            rows, cols = map(int, size.split('x'))
            break
        except:
            print("Invalid format.")
    print(f"\nEnter the elements for {name} row by row (space-separated):")
    elements = []
    for i in range(rows):
        while True:
            row_input = input(f"Row {i+1}: ").strip()
            row = row_input.split()
            if len(row) != cols:
                print(f"Invalid! Enter exactly {cols} elements.")
                continue
            try:
                row = [float(x) for x in row]
                break
            except:
                print("Invalid input. Please enter numbers only.")
        elements.append(row)
    return np.array(elements)

def display_matrix(matrix, name="Matrix"):
    print(f"\n{name}:")
    table = tabulate(matrix, tablefmt="grid", floatfmt=".2f")
    print(table)

def matrix_addition(A, B):
    if A.shape != B.shape:
        print("Error: Matrices must have the same dimensions for addition.")
        return None
    return A + B

def matrix_subtraction(A, B):
    if A.shape != B.shape:
        print("Error: Matrices must have the same dimensions for subtraction.")
        return None
    return A - B

def matrix_multiplication(A, B):
    if A.shape[1] != B.shape[0]:
        print("Error: Columns of A must equal rows of B for multiplication.")
        return None
    return np.dot(A, B)

def matrix_transpose(A):
    return A.T

def matrix_determinant(A):
    if A.shape[0] != A.shape[1]:
        print("Error: Determinant can only be calculated for square matrices.")
        return None
    return np.linalg.det(A)

def main():
    print("\n=== Matrix Operations Tool ===")
    while True:
        print("\nChoose an operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Transpose")
        print("5. Determinant")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ")

        if choice in ['1', '2', '3']:
            A = get_matrix("Matrix A")
            B = get_matrix("Matrix B")
            display_matrix(A, "Matrix A")
            display_matrix(B, "Matrix B")

            if choice == '1':
                result = matrix_addition(A, B)
                if result is not None:
                    display_matrix(result, "A + B")
            elif choice == '2':
                result = matrix_subtraction(A, B)
                if result is not None:
                    display_matrix(result, "A - B")
            elif choice == '3':
                result = matrix_multiplication(A, B)
                if result is not None:
                    display_matrix(result, "A x B")

        elif choice == '4':
            A = get_matrix("Matrix")
            display_matrix(A, "Original Matrix")
            result = matrix_transpose(A)
            display_matrix(result, "Transpose")

        elif choice == '5':
            A = get_matrix("Matrix")
            display_matrix(A, "Matrix")
            result = matrix_determinant(A)
            if result is not None:
                print(f"\nDeterminant: {result:.4f}")

        elif choice == '6':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
