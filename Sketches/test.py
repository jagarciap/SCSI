import mesh
import pic
import Species.species
import numpy

trackers = numpy.asarray([0, 26, 4001, 72, 900, 136, 4, 1000, 15, 901])
ind = numpy.asarray([0, 5, 61, 6, 124, 136, 1000, 125, 126, 4])
ind1 = numpy.sort(ind)
ind_trackers = numpy.argsort(trackers)
trackers1 = trackers[ind_trackers]

ind_c = 0
tracker_c = 0
n = 0
while ind_c != len(ind1) and tracker_c != len(trackers1):
    if ind1[ind_c] < trackers1[tracker_c]:
        n += 1
        ind_c += 1
        continue
    elif ind1[ind_c] > trackers1[tracker_c]:
        trackers1[tracker_c] -= n
        tracker_c += 1
        continue
    elif ind1[ind_c] == trackers1[tracker_c]:
        trackers1[tracker_c] = len(ind1)
        n += 1
        tracker_c += 1
        ind_c += 1

if ind_c == len(ind1) and tracker_c < len(trackers1):
    trackers1[tracker_c:] -= n

print(trackers1)
trackers[ind_trackers] = trackers1
print(trackers, tracker_c, ind_c, n)
