import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

root = Tk()
file_path = ''
Percentage_data = tk.DoubleVar()
Num_Cluster = tk.IntVar()


class KMeans:
    def __init__(self, num_clusters, max_iterations=100):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations

    def fit(self, X, df):
        # Randomly initialize centroids
        centroids = X[np.random.choice(X.shape[0], self.num_clusters, replace=False)]

        for _ in range(self.max_iterations):
            # Assign each data point to the nearest centroid
            labels = self._assign_clusters(X, centroids)

            # Update centroids based on the mean of data points in each cluster
            new_centroids = self._update_centroids(X, labels)

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.labels_ = labels
        self.cluster_centers_ = centroids

        # Detect outliers
        self.detect_outliers(X, centroids, df)  # Pass 'df' as the third argument

    def _assign_clusters(self, X, centroids):
        distances = np.abs(X - centroids[:, np.newaxis])
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.num_clusters, X.shape[1]))
        for i in range(self.num_clusters):
            cluster_points = X[labels == i]
            if cluster_points.shape[0] > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = X[np.random.choice(X.shape[0], 1)]
        return new_centroids

    def detect_outliers(self, X, centroids, df):
        Result = ''
        # Calculate distances of each point from its centroid
        distances = np.abs(X - centroids[self.labels_])
        max_distances = np.max(distances, axis=0)

        # Define outliers as points that are more than 3 standard deviations away from the mean distance
        outlier_indices = np.where(max_distances > np.mean(max_distances) + 3 * np.std(max_distances))[0]

        # Print outliers
        if len(outlier_indices) > 0:
            Result = Result + "\nOutlier movies:\n"
            for idx in outlier_indices:
                # Extract the film name from the DataFrame
                film_name = df.iloc[idx]['Movie Name']
                film_rate=df.iloc[idx]['IMDB Rating']
                Result = Result + f"{film_name}  : {film_rate} \n"
        else:
            Result = Result + "\nNo outlier movies.\n"
        result_text1.insert(tk.END, Result)


def visualize_clusters(X, labels, centroids, filename="cluster_plot.png"):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(np.unique(labels))):
        plt.scatter(X[labels == i], np.zeros_like(X[labels == i]), c=colors[i], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]), marker='*', s=200, c='k', label='Centroids')
    plt.xlabel('IMDB Rating')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.savefig(filename)  # Save the plot as an image with the specified filename



def browse_file():
    global file_path
    file_path = filedialog.askopenfilename()


def analyze_data():
    global file_path
    global Percentage_data
    global Num_Cluster

    df = pd.read_csv(file_path)

    # Prompt user for the percentage of data and the number of clusters (K)
    percentage = Percentage_data.get()
    k = Num_Cluster.get()

    # Calculate the number of records based on the percentage provided
    num_records = int(len(df) * percentage / 100)

    # Randomly select the calculated number of records from the dataset
    df = df.sample(n=num_records, random_state=42)
    # Reshape the data to a 2D array (required by sklearn KMeans)
    x = df['IMDB Rating'].values.reshape(-1, 1)

    # Instantiate and fit the KMeans model
    kmeans = KMeans(num_clusters=k)
    kmeans.fit(x, df)

    # Print the content of each cluster
    Result1 = ''
    for cluster_id in range(k):
        Result1 = Result1 + f"\nCluster {cluster_id + 1} movies:\n"
        cluster_movies = df[kmeans.labels_ == cluster_id]
        for _, row in cluster_movies.iterrows():
            film_name = row['Movie Name']  # Replace 'Film Name' with the actual column name containing film names
            film_rate=row['IMDB Rating']
            Result1 = Result1 + f"{film_name}  : {film_rate} \n"
    result_text1.insert(tk.END, Result1)

    # Visualize clusters and centroids
    visualize_clusters(x, kmeans.labels_, kmeans.cluster_centers_)
    # Load the generated graph image
    img1 = Image.open("C:\\Users\\DELL\\Downloads\\Assignment2\\cluster_plot.png")
    img1 = img1.resize((800, 600), Image.ANTIALIAS)
    img1 = ImageTk.PhotoImage(img1)
    image_label.configure(image=img1)
    image_label.image = img1




# Load the generated graph image
img = Image.open("C:\\Users\\DELL\\Downloads\\Assignment2\\result_graph.PNG")
img = img.resize((800, 600), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
# Create a Tkinter window
root.title("IMDB Clustering")

# Set the background color
root.configure(bg='#F699CD')

# Set the size of the root window
root.geometry('1460x1000')

# Disable resizing
root.resizable(False, False)

# Frame for file selection
frame_file = Frame(root)
frame_file.grid(row=0, column=0, padx=10, pady=10)

label_path = Label(frame_file, text="Select CSV file:")
label_path.grid(row=0, column=0)

entry_path = Entry(frame_file, width=50)
entry_path.grid(row=0, column=1, padx=10)

button_browse = Button(frame_file, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2)

# Frame for minimum support input
frame_support = Frame(root)
frame_support.grid(row=1, column=0, padx=10, pady=10)

label_Num_Cluster = Label(frame_support, text="Enter Number of Cluster:")
label_Num_Cluster.grid(row=0, column=0)

entry_Num_Cluster = Entry(frame_support, textvariable=Num_Cluster, width=10)
entry_Num_Cluster.grid(row=0, column=1, padx=10)

label_Percentage_record = Label(frame_support, text="Percentage of Record")
label_Percentage_record.grid(row=1, column=0)

entry_Percentage_record = Entry(frame_support, textvariable=Percentage_data, width=10)
entry_Percentage_record.grid(row=1, column=1, padx=10)

# Analyze button
button_analyze = Button(root, text="Analyze Data", command=analyze_data)
button_analyze.grid(row=2, column=0, pady=10)

result_frame = tk.Frame(root)
result_frame.grid(row=3, column=0, padx=10, pady=10)

result_label = tk.Label(result_frame, text="Clustering")
result_label.grid(row=0, column=0, padx=10, pady=10)

result_text1 = ScrolledText(result_frame, height=35, width=60)
result_text1.grid(row=0, column=1, padx=10, pady=10)

# Display the image on a label
image_label = Label(result_frame, image=img)
image_label.grid(row=0, column=2, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()