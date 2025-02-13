import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/Justin/OneDrive/Desktop/Data Folder/Customer Purchase Data.csv"
df = pd.read_csv(file_path)


# Function: Top 10 Customers by Spending Score vs Income
def spending_vs_income():
    # Get top 10 customers by Spending Score
    top_spenders = df.nlargest(10, "Spending_Score")

    # Get top 10 customers by Income
    top_members = df.nlargest(10, "Income")

    # Set up the figure for side-by-side plots with a wider size
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Increased figure width

    # Spending Score Plot
    sns.barplot(x="Number", y="Spending_Score", data=top_spenders, hue="Number", palette="viridis", legend=False,
                ax=axes[0])
    axes[0].set_title("Top 10 Customers by Spending Score")
    axes[0].set_xlabel("Customer Number")
    axes[0].set_ylabel("Spending Score")
    axes[0].tick_params(axis='x', rotation=45)

    # Add data labels for Spending Score
    for index, row in top_spenders.iterrows():
        axes[0].text(index, row["Spending_Score"] + 1, str(round(row["Spending_Score"], 1)), ha='center', fontsize=10)

    # Income Plot
    sns.barplot(x="Number", y="Income", data=top_members, hue="Number", palette="magma", legend=False, ax=axes[1])
    axes[1].set_title("Top 10 Customers by Income")
    axes[1].set_xlabel("Customer Number")
    axes[1].set_ylabel("Income")
    axes[1].tick_params(axis='x', rotation=45)

    # Add data labels for Income
    for index, row in top_members.iterrows():
        axes[1].text(index, row["Income"] + 0.2, str(round(row["Income"], 1)), ha='center', fontsize=10)

    # Adjust layout with extra padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Show the plots
    plt.show()


# Function: Top 10 Customers by Spending Score vs Membership Years
def spending_vs_membership():
    # Get top 10 customers by Spending Score
    top_spenders = df.nlargest(10, "Spending_Score")

    # Get top 10 customers by Membership Years
    top_members = df.nlargest(10, "Membership_Years")

    # Set up the figure for side-by-side plots with a wider size
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Increased figure width

    # Spending Score Plot
    sns.barplot(x="Number", y="Spending_Score", data=top_spenders, hue="Number", palette="viridis", legend=False,
                ax=axes[0])
    axes[0].set_title("Top 10 Customers by Spending Score")
    axes[0].set_xlabel("Customer Number")
    axes[0].set_ylabel("Spending Score")
    axes[0].tick_params(axis='x', rotation=45)

    # Add data labels for Spending Score
    for index, row in top_spenders.iterrows():
        axes[0].text(index, row["Spending_Score"] + 1, str(round(row["Spending_Score"], 1)), ha='center', fontsize=10)

    # Membership Years Plot
    sns.barplot(x="Number", y="Membership_Years", data=top_members, hue="Number", palette="magma", legend=False,
                ax=axes[1])
    axes[1].set_title("Top 10 Customers by Membership Years")
    axes[1].set_xlabel("Customer Number")
    axes[1].set_ylabel("Membership Years")
    axes[1].tick_params(axis='x', rotation=45)

    # Add data labels for Membership Years
    for index, row in top_members.iterrows():
        axes[1].text(index, row["Membership_Years"] + 0.2, str(round(row["Membership_Years"], 1)), ha='center',
                     fontsize=10)

    # Adjust layout with extra padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Show the plots
    plt.show()


# Function: Top 10 Customers by Spending Score vs Purchase Frequency
def spending_vs_frequency():
    # Get top 10 customers by Spending Score
    top_spenders = df.nlargest(10, "Spending_Score")

    # Get top 10 customers by purchase_frequency
    top_members = df.nlargest(10, "Purchase_Frequency")

    # Set up the figure for side-by-side plots with a wider size
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Increased figure width

    # Spending Score Plot
    sns.barplot(x="Number", y="Spending_Score", data=top_spenders, hue="Number", palette="viridis", legend=False,
                ax=axes[0])
    axes[0].set_title("Top 10 Customers by Spending Score")
    axes[0].set_xlabel("Customer Number")
    axes[0].set_ylabel("Spending Score")
    axes[0].tick_params(axis='x', rotation=45)

    # Add data labels for Spending Score
    for index, row in top_spenders.iterrows():
        axes[0].text(index, row["Spending_Score"] + 1, str(round(row["Spending_Score"], 1)), ha='center', fontsize=10)

    # purchase_frequency Plot
    sns.barplot(x="Number", y="Purchase_Frequency", data=top_members, hue="Number", palette="magma", legend=False,
                ax=axes[1])
    axes[1].set_title("Top 10 Customers by Purchase Frequency")
    axes[1].set_xlabel("Customer Number")
    axes[1].set_ylabel("Purchase Frequency")
    axes[1].tick_params(axis='x', rotation=45)

    # Add data labels for purchase_frequency
    for index, row in top_members.iterrows():
        axes[1].text(index, row["Purchase_Frequency"] + 0.2, str(round(row["Purchase_Frequency"], 1)), ha='center',
                     fontsize=10)

    # Adjust layout with extra padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Show the plots
    plt.show()


# Function: Top 10 Customers by Spending Score Only
def top_10_spenders():
    top_spenders = df.nlargest(10, "Spending_Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_spenders["Number"].astype(str), y=top_spenders["Spending_Score"], palette="coolwarm")

    plt.title("Top 10 Customers by Spending Score")
    plt.xlabel("Customer Number")
    plt.ylabel("Spending Score")
    plt.xticks(rotation=45)

    plt.show()


# Menu to Select Analysis
def main():
    print("\nChoose an Analysis to Run:")
    print("1. Spending Score vs Income")
    print("2. Spending Score vs Membership Years")
    print("3. Spending Score vs Purchase Frequency")
    print("4. Top 10 Customers by Spending Score")
    print("5. Run All Analyses")
    print("6. Exit")

    choice = input("Enter your choice (1-6): ")

    if choice == "1":
        spending_vs_income()
    elif choice == "2":
        spending_vs_membership()
    elif choice == "3":
        spending_vs_frequency()
    elif choice == "4":
        top_10_spenders()
    elif choice == "5":
        spending_vs_income()
        spending_vs_membership()
        spending_vs_frequency()
        top_10_spenders()
    elif choice == "6":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please select a valid option.")

    main()  # Re-run the menu after selection


# Run the program
if __name__ == "__main__":
    main()
