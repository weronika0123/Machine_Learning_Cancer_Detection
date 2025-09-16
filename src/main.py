from data_loader import load_data
from preprocessing import preprocess_data


def main():
    print("=== Program konsolowy szkielet ===")

    user_input = input("Podaj argument: ")

    data = load_data(user_input)
    if user_input == "1":
        data = preprocess_data(user_input)
        result = "wybrałaś 1"
    else:
        result = "wybrałaś cos innego"

    print("Result: ", result)

if __name__ == "__main__":
    main()