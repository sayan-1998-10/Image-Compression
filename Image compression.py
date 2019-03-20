""" K-means clustering """
import numpy as np
from skimage.io import imshow
from matplotlib import pyplot as plt 
import matplotlib.image as img
from  mpl_toolkits.mplot3d import Axes3D




x_centroid1 = np.zeros((16384,1))
x_centroid2 = np.zeros((16384,1))
x_centroid3 = np.zeros((16384,1))

x_centroid4 = np.zeros((16384,1))
x_centroid5 = np.zeros((16384,1))
x_centroid6 = np.zeros((16384,1))

x_centroid7 = np.zeros((16384,1))
x_centroid8 = np.zeros((16384,1))
x_centroid9 = np.zeros((16384,1))

x_centroid10 = np.zeros((16384,1))
x_centroid11 = np.zeros((16384,1))
x_centroid12 = np.zeros((16384,1))
x_centroid13 = np.zeros((16384,1))
x_centroid14= np.zeros((16384,1))
x_centroid15 = np.zeros((16384,1))
x_centroid16= np.zeros((16384,1))

c = np.zeros((16384,1))

def assign_centroid(X,centroids):
   

    for i in range(len(X)):
        #calculate the L2 norm
        dist_cent1 =  (X[i][1]-centroids[0][1])**2+(X[i][0]-centroids[0][0])**2 +(X[i][2]-centroids[0][2])**2
        dist_cent2 =  (X[i][1]-centroids[1][1])**2+(X[i][0]-centroids[1][0])**2 +(X[i][2]-centroids[1][2])**2
        dist_cent3 =  (X[i][1]-centroids[2][1])**2+(X[i][0]-centroids[2][0])**2 +(X[i][2]-centroids[2][2])**2
        dist_cent4 =  (X[i][1]-centroids[3][1])**2+(X[i][0]-centroids[3][0])**2 +(X[i][2]-centroids[3][2])**2
        dist_cent5 =  (X[i][1]-centroids[4][1])**2+(X[i][0]-centroids[4][0])**2 +(X[i][2]-centroids[4][2])**2
        dist_cent6 =  (X[i][1]-centroids[5][1])**2+(X[i][0]-centroids[5][0])**2 +(X[i][2]-centroids[5][2])**2
        dist_cent7 =  (X[i][1]-centroids[6][1])**2+(X[i][0]-centroids[6][0])**2 +(X[i][2]-centroids[6][2])**2
        dist_cent8 =  (X[i][1]-centroids[7][1])**2+(X[i][0]-centroids[7][0])**2 +(X[i][2]-centroids[7][2])**2
        dist_cent9 =  (X[i][1]-centroids[8][1])**2+(X[i][0]-centroids[8][0])**2 +(X[i][2]-centroids[8][2])**2
        dist_cent10=  (X[i][1]-centroids[9][1])**2+(X[i][0]-centroids[9][0])**2 +(X[i][2]-centroids[9][2])**2
        dist_cent11=  (X[i][1]-centroids[10][1])**2+(X[i][0]-centroids[10][0])**2 +(X[i][2]-centroids[10][2])**2
        dist_cent12=  (X[i][1]-centroids[11][1])**2+(X[i][0]-centroids[11][0])**2 +(X[i][2]-centroids[11][2])**2
        dist_cent13=  (X[i][1]-centroids[12][1])**2+(X[i][0]-centroids[12][0])**2 +(X[i][2]-centroids[12][2])**2
        dist_cent14=  (X[i][1]-centroids[13][1])**2+(X[i][0]-centroids[13][0])**2 +(X[i][2]-centroids[13][2])**2
        dist_cent15=  (X[i][1]-centroids[14][1])**2+(X[i][0]-centroids[14][0])**2 +(X[i][2]-centroids[14][2])**2
        dist_cent16=  (X[i][1]-centroids[15][1])**2+(X[i][0]-centroids[15][0])**2 +(X[i][2]-centroids[15][2])**2

        
        minimum = min(dist_cent1,dist_cent2,dist_cent3,dist_cent4,dist_cent5,dist_cent6,dist_cent7
        ,dist_cent8,dist_cent9,dist_cent10, dist_cent11,dist_cent12,dist_cent13,dist_cent14,dist_cent15
        ,dist_cent16)
        
    
        if (minimum == dist_cent1):
            c[i] = 1
        elif (minimum == dist_cent2):
            c[i] = 2
        elif (minimum == dist_cent3):
            c[i] = 3
        elif (minimum == dist_cent4):
            c[i] = 4
        elif (minimum == dist_cent5):
            c[i] = 5
        elif (minimum == dist_cent6):
            c[i] = 6
        elif (minimum == dist_cent7):
            c[i] = 7
        elif (minimum == dist_cent8):
            c[i] = 8
        elif (minimum == dist_cent9):
            c[i] = 9
        elif (minimum == dist_cent10):
            c[i] = 10
        elif (minimum == dist_cent11):
            c[i] = 11
        elif (minimum == dist_cent12):
            c[i] = 12
        elif (minimum == dist_cent13):
            c[i] = 13
        elif (minimum == dist_cent14):
            c[i] = 14
        elif (minimum == dist_cent15):
            c[i] = 15
        else:
            c[i] = 16
            
    
    return c
        
def position_ofCentroid(X,c):
    #stroinf the indexes where c[i]=1,2,3,....in x_centroid1,x_centroid2,....
    for i in range(len(c)):
        if  (c[i]==1):
            x_centroid1[i] = i
        elif (c[i]==2):
            x_centroid2[i] = i
        elif(c[i]==3):
            x_centroid3[i]= i
        elif(c[i]==4):
            x_centroid4[i]= i
        elif(c[i]==5):
            x_centroid5[i]= i
        elif(c[i]==6):
            x_centroid6[i]= i
        elif(c[i]==7):
            x_centroid7[i]= i
        elif(c[i]==8):
            x_centroid8[i]= i
        elif(c[i]==9):
            x_centroid9[i]= i
        elif(c[i]==10):
            x_centroid10[i]= i
        elif(c[i]==11):
            x_centroid11[i]= i
        elif(c[i]==12):
            x_centroid12[i]= i
        elif(c[i]==13):
            x_centroid13[i]= i
        elif(c[i]==14):
            x_centroid14[i]= i
        elif(c[i]==15):
            x_centroid15[i]= i
        else:
            x_centroid16[i]= i
   
    new_centroidPosition = np.zeros((16,3))
    
    #removing all the zeros from x_centroid'n'
    cluster1_points=(x_centroid1[x_centroid1!=0])
    cluster2_points=(x_centroid2[x_centroid2!=0])
    cluster3_points=(x_centroid3[x_centroid3!=0])
    cluster4_points=(x_centroid4[x_centroid4!=0])
    cluster5_points=(x_centroid5[x_centroid5!=0])
    cluster6_points=(x_centroid6[x_centroid6!=0])
    cluster7_points=(x_centroid7[x_centroid7!=0])
    cluster8_points=(x_centroid8[x_centroid8!=0])
    cluster9_points=(x_centroid9[x_centroid9!=0])
    cluster10_points=(x_centroid10[x_centroid10!=0])
    cluster11_points=(x_centroid11[x_centroid11!=0])
    cluster12_points=(x_centroid12[x_centroid12!=0])
    cluster13_points=(x_centroid13[x_centroid13!=0])
    cluster14_points=(x_centroid14[x_centroid14!=0])
    cluster15_points=(x_centroid15[x_centroid15!=0])
    cluster16_points=(x_centroid16[x_centroid16!=0])
    
    
    #define 16 empry arrays(full of zeros) for positions of 16 clusters
    new_pos1=np.array([0.0,0.0,0.0])
    new_pos2=np.array([0.0,0.0,0.0])
    new_pos3=np.array([0.0,0.0,0.0])
    new_pos4=np.array([0.0,0.0,0.0])
    new_pos5=np.array([0.0,0.0,0.0])
    new_pos6=np.array([0.0,0.0,0.0])
    new_pos7=np.array([0.0,0.0,0.0])
    new_pos8=np.array([0.0,0.0,0.0])
    new_pos9=np.array([0.0,0.0,0.0])
    new_pos10=np.array([0.0,0.0,0.0])
    new_pos11=np.array([0.0,0.0,0.0])
    new_pos12=np.array([0.0,0.0,0.0])
    new_pos13=np.array([0.0,0.0,0.0])
    new_pos14=np.array([0.0,0.0,0.0])
    new_pos15=np.array([0.0,0.0,0.0])
    new_pos16=np.array([0.0,0.0,0.0])
    
    
    #calculate the average of assigned points for each cluster
    for i in range(len(cluster1_points)):
        new_pos1+=X[int(cluster1_points[i])]
        
    for i in range(len(cluster2_points)):
        new_pos2+=X[int(cluster2_points[i])]
        
    for i in range(len(cluster3_points)):
        new_pos3+=X[int(cluster3_points[i])]
    
    for i in range(len(cluster4_points)):
        new_pos4+=X[int(cluster4_points[i])]
    
    for i in range(len(cluster5_points)):
        new_pos5+=X[int(cluster5_points[i])]
    
    for i in range(len(cluster6_points)):
        new_pos6+=X[int(cluster6_points[i])]
    
    for i in range(len(cluster7_points)):
        new_pos7+=X[int(cluster7_points[i])]
    for i in range(len(cluster8_points)):
        new_pos8+=X[int(cluster8_points[i])]
    
    for i in range(len(cluster9_points)):
        new_pos9+=X[int(cluster9_points[i])]
    for i in range(len(cluster10_points)):
        new_pos10+=X[int(cluster10_points[i])]
    
    for i in range(len(cluster11_points)):
        new_pos11+=X[int(cluster11_points[i])]
    
    for i in range(len(cluster12_points)):
        new_pos12+=X[int(cluster12_points[i])]
    for i in range(len(cluster13_points)):
        new_pos13+=X[int(cluster13_points[i])]
    for i in range(len(cluster14_points)):
        new_pos14+=X[int(cluster14_points[i])]
    for i in range(len(cluster15_points)):
        new_pos15+=X[int(cluster15_points[i])]
    for i in range(len(cluster16_points)):
        new_pos16+=X[int(cluster16_points[i])]
    
    
    new_pos1 = new_pos1/len(cluster1_points)
    new_pos2 = new_pos2/len(cluster2_points)
    new_pos3 = new_pos3/len(cluster3_points)
    new_pos4 = new_pos4/len(cluster4_points)
    new_pos5 = new_pos5/len(cluster5_points)
    new_pos6 = new_pos6/len(cluster6_points)
    new_pos7 = new_pos7/len(cluster7_points)
    new_pos8 = new_pos8/len(cluster8_points)
    new_pos9 = new_pos9/len(cluster9_points)
    new_pos10 = new_pos10/len(cluster10_points)
    new_pos11 = new_pos11/len(cluster11_points)
    new_pos12 = new_pos12/len(cluster12_points)
    new_pos13 = new_pos13/len(cluster13_points)
    new_pos14 = new_pos14/len(cluster14_points)
    new_pos15 = new_pos15/len(cluster15_points)
    new_pos16 = new_pos16/len(cluster16_points)


    #storing the position of new cluster centers in new_centroidPosition array    
    new_centroidPosition[0] = new_pos1
    new_centroidPosition[1] = new_pos2
    new_centroidPosition[2] = new_pos3
    new_centroidPosition[3] = new_pos4
    new_centroidPosition[4] = new_pos5
    new_centroidPosition[5] = new_pos6
    new_centroidPosition[6] = new_pos7
    new_centroidPosition[7] = new_pos8
    new_centroidPosition[8] = new_pos9
    new_centroidPosition[9] = new_pos10
    new_centroidPosition[10] = new_pos11
    new_centroidPosition[11] = new_pos12
    new_centroidPosition[12] = new_pos13
    new_centroidPosition[13] = new_pos14
    new_centroidPosition[14] = new_pos15
    new_centroidPosition[15] = new_pos16
    
    

    return new_centroidPosition    

def init_Centroid(X):
    randindx = np.random.permutation(16384)
    centroids = X[randindx[0:16],:]
    return centroids
def plot(X,c,centroids):
    
    fig = plt.figure()
    ax  = Axes3D(fig)
    #plotting the assigned points 
    ax.scatter(X[:,0],X[:,1],X[:,2],c=c[:].ravel(),cmap='viridis',alpha=0.1)
    # plotting the final centroids
    ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],s=300,c='red',alpha=1,marker='*')
def compress(centroids,X,c):
       #setting positon of each pixel to its corresponding cluster point 
    for i in range(len(X)):
        if(c[i]==1):
            X[i] = centroids[0]
        elif(c[i]==2):
            X[i] = centroids[1]
        elif(c[i]==3):
            X[i] = centroids[2]
        elif(c[i]==4):
            X[i] = centroids[3]
        elif(c[i]==5):
            X[i] = centroids[4]
        elif(c[i]==6):
            X[i] = centroids[5]
        elif(c[i]==7):
            X[i] = centroids[6]
        elif(c[i]==8):
            X[i] = centroids[7]
        elif(c[i]==9):
            X[i] = centroids[8]
        elif(c[i]==10):
            X[i] = centroids[9]
        elif(c[i]==11):
            X[i] = centroids[10]
        elif(c[i]==12):
            X[i] = centroids[11]
        elif(c[i]==13):
            X[i] = centroids[12]
        elif(c[i]==14):
            X[i] = centroids[13]
        elif(c[i]==15):
            X[i] = centroids[14]
        else:
            X[i] = centroids[15]
    return X
        
def main():
    image=img.imread("bird_small.png")
    image_matrix= image.reshape(16384,3)
    centroids = init_Centroid(image_matrix)
   
    for i in range(10):

        c=assign_centroid(image_matrix,centroids)
        print(i)        
        centroids=position_ofCentroid(image_matrix,c)
            
#Uncomment the next line if you want to plot the 3D scatter plot
        
 #   plot(image_matrix,c,centroids)
   image_matrix = compress(centroids,image_matrix,c)
    
   imshow(image_matrix.reshape((128,128,3)))
if __name__== '__main__':
    main()  
