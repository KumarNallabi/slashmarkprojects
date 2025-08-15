import pandas as pd

TASKS_FILE = 'tasks.csv'
PRIORITY_ORDER = {'High': 1, 'Medium': 2, 'Low': 3}

def load_tasks():
    try:
        return pd.read_csv(TASKS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=['description', 'priority'])

def save_tasks(df):
    df.to_csv(TASKS_FILE, index=False)

def list_tasks(df):
    if df.empty:
        print("\nNo tasks found.\n")
    else:
        print("\nCurrent Tasks:")
        print(df.reset_index(drop=True).to_string(index=True))
        print()

def add_task(df):
    desc = input("Enter task description: ").strip()
    priority = input("Enter priority (High/Medium/Low): ").strip().capitalize()
    if priority not in PRIORITY_ORDER:
        print("Invalid priority. Task not added.\n")
        return df
    new_task = pd.DataFrame([[desc, priority]], columns=['description', 'priority'])
    df = pd.concat([df, new_task], ignore_index=True)
    print("Task added successfully.\n")
    return df

def remove_task(df):
    list_tasks(df)
    try:
        index = int(input("Enter the index of the task to remove: "))
        if 0 <= index < len(df):
            df = df.drop(index).reset_index(drop=True)
            print("Task removed successfully.\n")
        else:
            print("Invalid index.\n")
    except ValueError:
        print("Invalid input.\n")
    return df

def recommend_tasks(df):
    if df.empty:
        print("\nNo tasks to recommend.\n")
        return
    df['priority_rank'] = df['priority'].map(PRIORITY_ORDER)
    recommended = df.sort_values(by='priority_rank').drop(columns='priority_rank')
    print("\nRecommended Tasks (sorted by priority):")
    print(recommended.reset_index(drop=True).to_string(index=True))
    print()

def main():
    df = load_tasks()
    while True:
        print("Task Management Menu:")
        print("1. List tasks")
        print("2. Add task")
        print("3. Remove task")
        print("4. Recommend tasks")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ").strip()
        print()
        if choice == '1':
            list_tasks(df)
        elif choice == '2':
            df = add_task(df)
            save_tasks(df)
        elif choice == '3':
            df = remove_task(df)
            save_tasks(df)
        elif choice == '4':
            recommend_tasks(df)
        elif choice == '5':
            print("Exiting Task Management App. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()