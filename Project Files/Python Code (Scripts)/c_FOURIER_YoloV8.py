from b_MODEL_YoloV8 import *

#placing the extracted image around the origin of the complex plane
centroid = np.mean(edges)
edges = edges - centroid

#Get coordinates (x = fft_result.real, y = fft_result.imag) and phase (phase = np.angle(fft_result)) of the fourier coefficients
fft_result = np.fft.fft(edges)

#Get the corresponding frequencies of the fourier coefficients
frequencies = np.fft.fftfreq(len(edges))

#EXTRACTING RELEVANT PARAMETERS FROM FFT#

#Define empty lists for coordinates (vectors), magnitude & phase (phase list), and frequency (frequencies) of fourier coefficients
phase = []
vectors = []
frequency = []
magnitudes = []

#Keeping only the first few coefficients#
for i in range(len(fft_result)):
    if abs(fft_result[i]) > threshold: #coefficient threshold
        phase.append(np.angle(fft_result[i]))
        vectors.append([fft_result[i].real, fft_result[i].imag]) #used to define the complex exponentials as vectors. It is a convenient representation as the coordinates allow for easy definition of the vectors in manim (using Vector([x,y]), with the phase and magnitudes easily defined in C using only x and y)
        frequency.append(frequencies[i]) #used for determining how fast the complex exponential (vector) should rotate in the subsequent animation
        magnitudes.append(np.abs(fft_result[i])) #used for plotting the distribution of fourier coefficients in frequency-space

#CENTERING THE DATA (Frequency list & Vector List) AT THE 0 FREQUENCY COMPONENT#

data = np.array(np.fft.fftshift(vectors))
f = np.array(np.fft.fftshift(frequency))

#PLOTTING THE ORIGINAL SHAPE AND THE DISTRIBUTION OF COEFFICIENT MAGNITUDES VS FREQUENCY#

fig, ax = plt.subplots(figsize = (width/l1, height/h1))
ax.set_facecolor('white')
ax.scatter(edges.real, -edges.imag, color='black', s=0.2, linestyle = "dotted")
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linestyle('dotted')
ax.spines['bottom'].set_linestyle('dotted')
ax.set_xlim(min(edges.real) - 500, max(edges.real) + 500)
ax.set_ylim(min(edges.imag) - 500, max(edges.imag) + 500)
ax.set_title('Outline in Complex Plane', color='black', fontsize=10, pad=20, y = 1, x = 0.5)
plt.tick_params(axis='both', which='major', labelsize=8, colors='black')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f i'))
plt.show()

#Fourier coefficients
fig, ax = plt.subplots(figsize = (width/l, height/h))
ax.set_facecolor('white')
marker, stemline, baseline = ax.stem(f, magnitudes)
plt.setp(stemline, linewidth = 2)
plt.setp(marker, markersize = 0.01)
plt.setp(baseline, linewidth = 0.001)
ax.set_xlabel('Frequency', color = "black")
ax.set_ylabel('Magnitude', color = "black")
ax.set_title(str(len(f)) + " " + 'Total Fourier Coefficients', color = "Black")
ax.tick_params(axis='both', colors='black')
plt.show()