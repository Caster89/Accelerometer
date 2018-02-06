import numpy as np
import scipy as sp
import scipy.signal
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import os as os
import quaternion
import logging
logging.basicConfig(format='[@%(module)s.%(funcName)s] - %(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)




class Position(object):
    def __init__(self,filePath, kernel_dize=19, plot = False):
        #First column is the elapsed time and is saved in t
        self.t = np.genfromtxt(filePath,skip_header = 1,delimiter = ';',usecols=0)/1000
        #Columns 1-4 are the saved in the quaternion matrix
        self.quaternionMatrix = np.genfromtxt(filePath,skip_header = 1,delimiter = ';',usecols=(1,2,3,4))
        self.smoothQuaternionMatrix = np.empty((self.quaternionMatrix.shape))
        #The row values of the quaternion matrix are smoothed and then used to create the quaternion object
        #and create a vector
        sqw = sp.signal.medfilt(self.quaternionMatrix[:,0], kernel_size = 19)
        sqx= sp.signal.medfilt(self.quaternionMatrix[:,1], kernel_size = 19)
        sqy= sp.signal.medfilt(self.quaternionMatrix[:,2], kernel_size = 19)
        sqz= sp.signal.medfilt(self.quaternionMatrix[:,3], kernel_size = 19)
        self.quaternionVector = [quaternion.quaternion(w,x,y,z) for w,x,y,z in zip(sqw,sqx,sqy,sqz)]
        if plot:
            plt.figure()
            plt.plot(self.t,sqw,label = 'Raw quaternion w')
            plt.plot(self.t,sqx,label = 'Raw quaternion x')
            plt.plot(self.t,sqy,label = 'Raw quaternion y')
            plt.plot(self.t,sqz,label = 'Raw quaternion z')
            plt.legend()

        #Columns 5-7 are the values of acceleration
        self.accelVector = np.genfromtxt(filePath,skip_header = 1,delimiter = ';',usecols=(5,6,7))
        self.accelVector[:,0] = sp.signal.medfilt(self.accelVector[:,0], kernel_size = 51)
        self.accelVector[:,1] = sp.signal.medfilt(self.accelVector[:,1], kernel_size = 51)
        self.accelVector[:,2] = sp.signal.medfilt(self.accelVector[:,2], kernel_size = 51)
        if plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(self.t,self.accelVector[:,0],label = 'Raw accel x', color = 'tab:blue')
            ax1.plot(self.t,self.accelVector[:,1],label = 'Raw accell y', color = 'tab:blue')
            ax1.tick_params(color = 'tab:blue')
            plt.legend()
            ax2 = ax1.twinx()
            ax2.plot(self.t,self.accelVector[:,2],label = 'Raw gravity z', color = 'tab:red')
            ax2.tick_params(color = 'tab:red')

            plt.legend()
        #totGrav = self.accelVector**2
        #totGrav = totGrav.sum(axis = 1)
        #totGrav = np.sqrt(totGrav)
        #The intensity of gravity and the direction is calculated as an
        #as an average. The first up untill when the acceleration is constant
        #the second as an average of the first 100 points
        gravVect,totGrav = self.measureGravity(self.accelVector, 0.01)
        self.gravConv = 8.9/totGrav
        #calculate the quaternion to rotate the residual graviy
        #to 0,0,-1
        self.finalRot = self.createQuatFromVect(gravVect,[0.,0.,-1.])#self.createRotationQuat(gravVect,np.array([0.,0.,-1.]))
        #print 'final rotation is {}. Also calculated as:{}'.format(self.finalRot, self.createQuatFromVect(gravVect,[0.,0.,-1.]))
        if plot:
            plt.figure()
            #plt.plot(self.t,self.accelVector[:,0],label = 'Raw accel x')
            #plt.plot(self.t,self.accelVector[:,1],label = 'Raw accel y')
            #plt.plot(self.t,self.accelVector[:,2],label = 'Raw accel z')
            plt.plot(self.t,[totGrav]*len(self.t),label = 'Raw Gravity')
            plt.legend()
        #Create the vector containing the direction of gravity at all times
        #self.gravity =np.array([self.getGravity(q) for q in self.quaternionVector])
        self.gravity = np.array([quaternion.rotate_vectors(q, gravVect) for q in self.quaternionVector])
        self.gravity[:,0] = sp.signal.medfilt(self.gravity[:,0],3)
        self.gravity[:,1] = sp.signal.medfilt(self.gravity[:,1],3)
        self.gravity[:,2] = sp.signal.medfilt(self.gravity[:,2],3)

        #the gravity is rotated in order to have it point at 0,0,-1
        #self.gravity = np.array([rotate(g,self.finalRot) for g in self.gravity])
        #the accel vector i rotate by the same amount
        #self.accelVector = np.array([a,self.finalRot] for a in self.accelVector)
        if plot:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(self.t,self.gravity[:,0],label = 'Raw gravity x', color = 'tab:blue')
            ax1.plot(self.t,self.gravity[:,1],label = 'Raw gravity y', color = 'tab:blue')
            ax1.tick_params(color = 'tab:blue')
            plt.legend()
            ax2 = ax1.twinx()
            ax2.plot(self.t,self.gravity[:,2],label = 'Raw gravity z', color = 'tab:red')
            ax2.tick_params(color = 'tab:red')

            plt.legend()

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            fig.suptitle('Real Gravity')
            for n in range(100):
                ax.plot([0,self.gravity[n,0]],[0,self.gravity[n,1]],[0,self.gravity[n,2]])

        #Subtract the effect of gravity
        self.corrAccel = self.gravConv*np.array([quaternion.rotate_vectors(quat, lAcc) for lAcc,quat in zip(self.accelVector,self.quaternionVector)])-self.gravity*8.9
        if plot:
            plt.figure()
            plt.plot(self.t,self.corrAccel[:,0],label = 'Accel w/out gravity x')
            plt.plot(self.t,self.corrAccel[:,1],label = 'Accel w/out gravity y')
            plt.plot(self.t,self.corrAccel[:,2],label = 'Accel w/out gravity z')
            plt.legend()
        '''self.linAccel = self.accelVector - self.gravity*np.average(totGrav)
        if plot:
            plt.figure()
            plt.plot(self.t,self.linAccel[:,0],label = 'Accel w/out gravity x')
            plt.plot(self.t,self.linAccel[:,1],label = 'Accel w/out gravity y')
            plt.plot(self.t,self.linAccel[:,2],label = 'Accel w/out gravity z')
            plt.legend()'''

        #avrg = np.average(self.linAccel[:100,:],axis=0)
        #print 'Avereging the first 100 elements of the linear acceleration resulted in:',np.average(self.linAccel[:100,:],axis=0)
        #Remove from the linear acceleration the remaining background - this should be done by removing gravity
        #self.linAccel = self.linAccel-
        #Rotate the linear acceleration by the quaternion to obtain the real world acceleration direciton
        #self.accelRealWorld = np.array([quaternion.rotate_vectors(quat, lAcc) for lAcc,quat in zip(self.linAccel,self.quaternionVector)])
        self.accelRealWorld = np.array([quaternion.rotate_vectors(self.finalRot, aa) for aa in self.corrAccel])

        #self.accelRealWorld = np.array([self.rotate(lAcc,quat) for lAcc,quat in zip(self.linAccel,self.quaternionVector)])
        #self.accelRealWorld = np.array([self.rotate(aRW,self.finalRot) for aRW in self.accelRealWorld])
        if plot:
            plt.figure()
            plt.plot(self.t,self.accelRealWorld[:,0]/totGrav,label = 'True Accel x')
            plt.plot(self.t,self.accelRealWorld[:,1]/totGrav,label = 'True Accel y')
            plt.plot(self.t,self.accelRealWorld[:,2]/totGrav,label = 'True Accel z')
            plt.legend()

        vx = sp.integrate.cumtrapz(self.accelRealWorld[:,0], self.t, initial = 0)
        vy = sp.integrate.cumtrapz(self.accelRealWorld[:,1], self.t, initial = 0)
        vz = sp.integrate.cumtrapz(self.accelRealWorld[:,2], self.t, initial = 0)

        px = sp.integrate.cumtrapz(vx, self.t, initial = 0)
        py = sp.integrate.cumtrapz(vy, self.t, initial = 0)
        pz = sp.integrate.cumtrapz(vz, self.t, initial = 0)

        self.velocityVector = np.stack((vx,vy,vz), axis=-1)
        if plot:
            plt.figure()
            plt.plot(self.t,self.velocityVector[:,0]/totGrav,label = 'Speed x')
            plt.plot(self.t,self.velocityVector[:,1]/totGrav,label = 'Speed y')
            plt.plot(self.t,self.velocityVector[:,2]/totGrav,label = 'Speed z')
            plt.legend()

        self.positionVector = np.stack((px,py,pz), axis=-1)
        if plot:
            plt.figure()
            plt.plot(self.t,self.positionVector[:,0]/totGrav,label = 'Position x')
            plt.plot(self.t,self.positionVector[:,1]/totGrav,label = 'Position y')
            plt.plot(self.t,self.positionVector[:,2]/totGrav,label = 'Position z')
            plt.legend()

    def getGravity(self, q):
        #Return the direction of gravity in x,y,z coordinates
        #NOt sure of the mathematical reason
        x = 2*(q.x*q.z-q.w*q.y)
        y = 2*(q.w*q.x+q.y*q.z)
        z = (q.w*q.w-q.x*q.x-q.y*q.y+q.z*q.z)
        return np.array([x,y,z])

    def rotate(self,accel,q):

            q_cong = q.getConjugate()

            p = Quaternion(0,accel[0],accel[1],accel[2])

            p = q.getProduct(p)

            p = p.getProduct(q_cong)

            return np.array([p.x,p.y,p.z])

    def plotAccel(self):
        fig=plt.figure()
        fig.set_size_inches(10,7,forward=True)
        ax = fig.add_subplot(111)
        ax.plot(self.t,self.accelRealWorld[:,0],label='X')
        ax.plot(self.t,self.accelRealWorld[:,1],label='Y')
        ax.plot(self.t,self.accelRealWorld[:,2],label='Z')
        ax.legend(loc='best')

    def smooth(self,x,filterOrder = 3, filterCut = 0.01, window_len=11,window='hanning'):
        #from numpy import sin, cos, pi, linspace
        #from numpy.random import randn
        #from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

        #from matplotlib.pyplot import plot, legend, show, hold, grid, figure, savefig



        # Create an order 3 lowpass butterworth filter.
        b, a = sp.signal.butter(filterOrder, filterCut)

        # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
        # of the filter.
        zi = sp.signal.lfilter_zi(b, a)
        z, _ = sp.signal.lfilter(b, a, x, zi=zi*x[0])

        # Apply the filter again, to have a result filtered at an order
        # the same as filtfilt.
        z2, _ = sp.signal.lfilter(b, a, z, zi=zi*z[0])

        # Use filtfilt to apply the filter.
        y = sp.signal.filtfilt(b, a, x)
        return y

    def measureGravity(self,g, variance):
        cont = True
        i=3
        mag = np.sqrt(np.sum(g**2,axis = 1))
        while cont:
            c = np.average(mag[:i])
            err = np.sqrt(np.sum(((g[:i]-c)/g[:i])**2))

            if err>variance or i == mag.size-1:
                i-=1
                grav = np.average(mag[:i])
                cont = False
            i+=1
        dVect = np.average(g[:100,:],axis = 0)
        dVect = dVect/(np.linalg.norm(dVect))
        return (dVect,grav)

    def createRotationMat(self,a,b):
        theta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

        x = np.cross(a,b)/np.linalg.norm(np.cross(a,b))
        I=np.identity(3)
        A = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
        Rot =I+np.sin(theta)*A+(1-np.cos(theta))*A**2

    def animateCube(self,saveName):
        fig,ax = self.plotCube(1,4,1)
        #intervals= self.t[1:]-self.t[:-1]
        text = ax.text(0.1,0.8,0.8,'elev:{} azim:{}'.format(0,0),transform=ax.transAxes)
        #time.sleep(intervals[i])
        anim = animation.FuncAnimation(fig, self.animRotateAxis, frames=len(self.quaternionVector), fargs=(ax,text))
        anim.save(saveName, fps=30, extra_args=['-vcodec', 'libx264'])

    def animRotateAxis(self,i,ax,text):

        rot = p.rotate([1,0,0],self.quaternionVector[i])
        elev = np.rad2deg(np.arccos(rot[2]))
        azim = np.rad2deg(np.arcsin(rot[0]))
        ax.view_init(elev=elev, azim=azim)
        text.set_text('Quat:{},{},{},{}'.format(self.quaternionVector[i].w,self.quaternionVector[i].x,self.quaternionVector[i].y,self.quaternionVector[i].z))


    def plotCube(self,x,y,z):
        fig = plt.figure(figsize =[6,6])
        ax = fig.add_subplot(111, projection = '3d')
        #Face1
        x1 = np.array([[0,x,x,0,0],[0,0,0,0,0]])-x/2
        y1 = np.array([[0,0,0,0,0],[0,0,0,0,0]])-y/2
        z1 = np.array([[0,0,z,z,0],[0,0,0,0,0]])-z/2
        #Face2
        x2 = np.array([[0,0,0,0,0],[0,0,0,0,0]])-x/2
        y2 = np.array([[0,y,y,0,0],[0,0,0,0,0]])-y/2
        z2 = np.array([[0,0,z,z,0],[0,0,0,0,0]])-z/2
        #Face3
        x3 = np.array([[0,x,x,0,0],[0,0,0,0,0]])-x/2
        y3 = np.array([[0,0,y,y,0],[0,0,0,0,0]])-y/2
        z3 = np.array([[z,z,z,z,z],[z,z,z,z,z]])-z/2
        #Face4
        x4 = np.array([[0,x,x,0,0],[0,0,0,0,0]])-x/2
        y4 = np.array([[y,y,y,y,y],[y,y,y,y,y]])-y/2
        z4 = np.array([[0,0,z,z,0],[0,0,0,0,0]])-z/2
        #Face5
        x5 = np.array([[0,0,x,x,0],[0,0,0,0,0]])-x/2
        y5 = np.array([[0,y,y,0,0],[0,0,0,0,0]])-y/2
        z5 = np.array([[0,0,0,0,0],[0,0,0,0,0]])-z/2
        #Face6
        x6 = np.array([[x,x,x,x,x],[x,x,x,x,x]])-x/2
        y6 = np.array([[0,y,y,0,0],[0,0,0,0,0]])-y/2
        z6 = np.array([[0,0,z,z,0],[0,0,0,0,0]])-z/2

        ax.plot_surface(x1,y1,z1,color='r')
        ax.plot_surface(x2,y2,z2,color='b')
        ax.plot_surface(x3,y3,z3,color='g')
        ax.plot_surface(x4,y4,z4,color='r')
        ax.plot_surface(x5,y5,z5,color='g')
        ax.plot_surface(x6,y6,z6,color='b')
        maxSize = max([x,y,z])
        ax.set_zlim(-maxSize,maxSize)
        ax.set_ylim(-maxSize,maxSize)
        ax.set_xlim(-maxSize,maxSize)
        ax.view_init(elev=0,azim=0)
        ax.set_axis_off()
        return [fig,ax]


    def createQuatFromVect(self,a,b):
        a = a/np.linalg.norm(a)
        b = b/np.linalg.norm(b)
        v1 = np.array([0,0,0])
        #If the two vectors are parallel don't rotate
        if np.dot(a,b)>0.99999:
            return quaternion.quaternion(1,0,0,0).normalized()
        #if the two vectors are antiparallel rotate by 180
        elif np.dot(a,b)<-0.99999:
            rotAx = np.cross(np.array([1,0,0]), a)
            if(np.linalg.norm(rotAx) < 0.00001):
                rotAx = np.cross(np.array([0,1,0]), a)
            return quaternion.quaternion(np.pi,rotAx[0],rotAx[1],rotAx[2]).normalized
        else:
            v1 = np.cross(a,b)
            w = np.linalg.norm(a)*np.linalg.norm(b)+np.dot(a,b)
        return quaternion.quaternion(w,v1[0],v1[1], v1[2]).normalized()

    def createRotationQuat(self,a,b):
        abNorm = np.sqrt(np.dot(a, a) * np.dot(b, b))
        realPart = abNorm + np.dot(a, b);
        w = np.array([0,0,0])

        if (realPart < 1.e-6 * abNorm):
            #If u and v are exactly opposite, rotate 180 degrees
            #around an arbitrary orthogonal axis. Axis normalisation
            #can happen later, when we normalise the quaternion. */
            realPart = 0.0;
            w = np.array([-a[1],a[0],0]) if abs(a[0]>abs[2]) else np.array([0,-a[2],a[1]])

        else:
            #Otherwise, build quaternion the standard way. */
            w = np.cross(a, b);
        quat = quaternion.quaternion(realPart,w[0],w[1],w[2])
        quat.normalized()
        return quat
