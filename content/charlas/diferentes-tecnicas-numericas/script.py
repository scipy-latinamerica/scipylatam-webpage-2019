import numpy as np
import matplotlib.pylab as plt

########### Differentiation ##################

print('Differentiation')
print('\n')

def fun(x):
	return np.sin(x)

x = np.linspace(0,np.pi,1000)
y = fun(x)

plt.figure()
plt.plot(x,y)
plt.grid(1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('sin(x).png')

def fun_prime(x):
	return np.cos(x)

y_prime = fun_prime(x)

plt.figure()
plt.plot(x,y_prime)
plt.grid(1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('sin_prime(x).png')

def forward_difference(x,y):
	h = (max(x)-min(x))/float(len(x)-1)
	prime = (y[1:]-y[0:-1])/float(h)
	return prime

y_prime_forward = forward_difference(x,y)

plt.figure()
plt.plot(x[:-1],y_prime_forward)
plt.grid(1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('sin_prime_forward(x).png')

def backward_difference(x,y):
	h = (max(x)-min(x))/float(len(x)-1)
	prime = (y[1:]-y[0:-1])/float(h)
	return prime

y_prime_backward = backward_difference(x,y)

plt.figure()
plt.plot(x[1:],y_prime_backward)
plt.grid(1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('sin_prime_backward(x).png')

def central_difference(x,y):
	h = (max(x)-min(x))/float(len(x)-1)
	prime = (y[2:]-y[0:-2])/float(2*h)
	return prime

y_prime_central = central_difference(x,y)

plt.figure()
plt.plot(x[1:-1],y_prime_central)
plt.grid(1)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('sin_prime_central(x).png')

def complete_prime(x,y):			
	h = (max(x)-min(x))/float(len(x)-1)
	prime_0 = float(y[1]-y[0])/float(h)
	prime_last = float(y[-1]-y[-2])/float(h)
	prime = (y[2:]-y[0:-2])/float(2*h)
	complete_prime = np.concatenate([[prime_0],prime,[prime_last]])
	return complete_prime

print('Error associated with forward difference: ' + str(np.sum(np.square(np.subtract(y_prime[:-1],y_prime_forward))))) #Mean squared error
print('Error associated with backward difference: ' + str(np.sum(np.square(np.subtract(y_prime[1:],y_prime_backward)))))
print('Error associated with central difference: ' + str(np.sum(np.square(np.subtract(y_prime[1:-1],y_prime_central)))))
print('Error associated with complete difference: ' + str(np.sum(np.square(np.subtract(y_prime,complete_prime(x,y))))))
print('\n')

########### Integration ##################

from scipy import integrate

print('Integration')
print('\n')

def int_trap(x,y):
	h = (max(x)-min(x))/float(len(x)-1)
	y *= h
	integral = np.sum(y[1:-1]) + ((y[0]+y[-1])/2.0)
	return integral

trapezoids = int_trap(x,fun(x))
trapezoids_scipy = integrate.trapz(fun(x), x)

print('Integral of sin(x)[0,pi] using trapezoids:' + '\n' + 'Using our implementation: ' + str(trapezoids) + '\n' 'Using SciPy: ' +str(trapezoids_scipy))
print('\n')

x_simp = np.linspace(0,np.pi,1001)

def int_simpson(x,y):
	h = (max(x)-min(x))/float(len(x)-1)
	y *= h
	integral = np.sum(y[1:-1:2]*4.0/3.0) + np.sum(y[2:-2:2]*2.0/3.0) + ((y[0]+y[-1])/3.0)
	return integral

simpson = int_simpson(x_simp,fun(x_simp))
simpson_scipy = integrate.simps(fun(x_simp), x_simp)

print('Integral of sin(x)[0,pi] using Simpson\'s Method:' + '\n' + 'Using our implementation: ' + str(simpson) + '\n' 'Using SciPy: ' + str(simpson_scipy))
print('\n')

x_mc = np.linspace(0,2*np.pi,1000)
y_mc = fun(x_mc)
y_random = np.random.uniform(-1,1,1000)
plt.figure()
plt.scatter(x_mc,y_random)
plt.plot(x_mc,y_mc,c='r')
plt.plot(x_mc,np.zeros(len(x)),c='b')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(1)
plt.savefig('monte_carlo.png')

def int_mc(x_min,x_max,y,N):
	counter = []
	y_max = max(y)
	y_min = min(y)
	area = (x_max-x_min)*(y_max-y_min)
	y_ran = np.random.uniform(y_min,y_max,N)
	for i in range(N):
		if(y_ran[i]>0 and y[i]>0 and abs(y_ran[i])<=abs(y[i])):
			counter.append(1)
		elif(y_ran[i]<0 and y[i]<0 and abs(y_ran[i])<=abs(y[i])):
			counter.append(-1)
		else:
			counter.append(0)
	return (np.mean(counter)*area)

monte_carlo_1000 = int_mc(0,np.pi,fun(np.random.uniform(0,2*np.pi,1000)),1000)
monte_carlo_10000 = int_mc(0,np.pi,fun(np.random.uniform(0,2*np.pi,10000)),10000)
monte_carlo_100000 = int_mc(0,np.pi,fun(np.random.uniform(0,2*np.pi,100000)),100000)

print('Integral of sin(x)[0,2pi] using Monte Carlo\'s Method:' + '\n' + 'Using 1000 points: ' + str(monte_carlo_1000) + '\n' 'Using 10000 points: ' + str(monte_carlo_10000) + '\n' 'Using 100000 points: ' + str(monte_carlo_100000))
print('\n')

########### Filters ##################

from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal

our_signal = np.genfromtxt('signal.dat', delimiter = ',')

signal_x = our_signal[:,0]
signal_y = our_signal[:,1]

plt.figure()
plt.plot(signal_x,signal_y)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(1)
plt.savefig('noisy_signal.png')

fourier_transform = np.real(fft(signal_y))
frequencies = fftfreq(len(signal_x),signal_x[1]-signal_x[0])

plt.figure()
plt.plot(frequencies,fourier_transform)
plt.xlabel('$f$')
plt.ylabel('Amplitude')
plt.grid(1)
plt.savefig('fourier_transform.png')

def filter_lowpass(frequencies,transform,n):
	for i in range(0,len(frequencies)):
		if abs(frequencies[i])>n:
			transform[i] = 0

	return transform

def filter_highpass(frequencies,transform,n):
	for i in range(0,len(frequencies)):
		if abs(frequencies[i])<n:
			transform[i] = 0

	return transform

signal_y_lowpass = np.real(ifft(filter_lowpass(frequencies, fourier_transform, 1000)))
signal_y_highpass = np.real(ifft(filter_highpass(frequencies, fourier_transform, 1000)))

b_low, a_low = signal.butter(3, 1000/((1/(signal_x[1]-signal_x[0]))/2), 'low')
scipy_y_lowpass = signal.filtfilt(b_low, a_low, signal_y)

b_high, a_high = signal.butter(3, 1000/((1/(signal_x[1]-signal_x[0]))/2), 'high')
scipy_y_highpass = signal.filtfilt(b_high, a_high, signal_y)

plt.figure()
plt.plot(signal_x,signal_y_lowpass, label = 'Ours')
plt.plot(signal_x,scipy_y_lowpass, label = 'Scipy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(1)
plt.legend()
plt.savefig('lowpass.png')

plt.figure()
plt.plot(signal_x,scipy_y_highpass, label = 'Scipy')
plt.plot(signal_x,signal_y_highpass, label = 'Ours')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(1)
plt.savefig('highpass.png')


########### Last Integral ##################

def last_fun(x):
	fun = np.sin(x)/x
	fun[np.isnan(fun)] = 1
	return fun

x_last = np.linspace(-10**6,10**6,101)
fun_last = last_fun(x_last)

plt.figure()
plt.plot(x_last,fun_last)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(1)
plt.savefig('improper.png')

integral = int_simpson(np.linspace(-10**6,10**6,10**6 +1),last_fun(np.linspace(-10**6,10**6,10**6+1)))

print('The integral of sin(x)/x [-infinity,infinity] using numerical methods is: ' + str(integral))
print('With an error of ' + str(100*(np.pi-integral)/np.pi) + '%')

########### Example ##################

boat_data = np.genfromtxt('boat_data.txt')

boat_t = boat_data[:,0]
boat_acce = boat_data[:,1]
boat_pitch = boat_data[:,2]

plt.figure()
plt.plot(boat_t,boat_acce)
plt.xlabel('t $(s)$')
plt.ylabel('A $(m/s^2)$')
plt.grid(1)
plt.savefig('acce_unfiltered.png')

plt.figure()
plt.plot(boat_t,boat_pitch)
plt.xlabel('t $(s)$')
plt.ylabel('P $(^\circ/s)$')
plt.grid(1)
plt.savefig('pitch_unfiltered.png')

a_acce, b_acce = signal.butter(2,(0.27/50), 'low')
a_pitch, b_pitch = signal.butter(2,(0.6341/50), 'low')

acce_filt = signal.filtfilt(a_acce,b_acce,boat_acce)
pitch_filt = signal.filtfilt(a_pitch,b_pitch,boat_pitch)

plt.figure()
plt.plot(boat_t,acce_filt)
plt.xlabel('t $(s)$')
plt.ylabel('A $(m/s^2)$')
plt.grid(1)
plt.savefig('acce_filtered.png')

plt.figure()
plt.plot(boat_t,pitch_filt)
plt.xlabel('t $(s)$')
plt.ylabel('P $(^\circ/s)$')
plt.grid(1)
plt.savefig('pitch_filtered.png')

def cum_trapz(x,d_y,y_0):
	y = np.empty(len(x))
	y[0] = y_0
	for i in range(0,len(x)-1):
		y[i+1] = (x[i+1]-x[i])*(((d_y[i+1]+d_y[i]))/2.0)+y[i]
	return y

x_boat = cum_trapz(boat_t,cum_trapz(boat_t,acce_filt,0),0)
angle_boat = cum_trapz(boat_t,pitch_filt,0)

plt.figure()
plt.plot(boat_t,x_boat)
plt.xlabel('t $(s)$')
plt.ylabel('x $(m)$')
plt.grid(1)
plt.savefig('x_boat.png')

plt.figure()
plt.plot(boat_t,angle_boat)
plt.xlabel('t $(s)$')
plt.ylabel('Angle $(^\circ)$')
plt.grid(1)
plt.savefig('angle_boat.png')