package org.vitrivr.cineast.core.util.mesh;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.joml.Vector3f;
import org.vitrivr.cineast.core.features.ClusterD2AndColorHistogram.FeatureSample;

public class IsodataClustering {
	public static class Cluster {
		public final Vector3f center = new Vector3f();
		public final List<FeatureSample> samples = new ArrayList<>();
		public float meanDistance; //d

		public void setCenter(Vector3f center) {
			this.center.x = center.x;
			this.center.y = center.y;
			this.center.z = center.z;
		}

		public void clear() {
			for(FeatureSample sample : this.samples) {
				if(sample.getCluster() == this) {
					sample.setCluster(null);
				}
			}
			this.samples.clear();
		}
	}

	private final int minSamplesPerCluster; //Pn
	private final float standardDeviation; //Ps
	private final float mergeDistance; //Pc
	private final int minIterations;
	private final int maxIterations; //I
	private final int nonConvergenceCutOff;

	public IsodataClustering(int minSamplesPerCluster, float standardDeviation, float mergeDistance, int minIterations, int maxIterations, int nonConvergenceCutOff) {
		this.minSamplesPerCluster = minSamplesPerCluster;
		this.standardDeviation = standardDeviation;
		this.mergeDistance = mergeDistance;
		this.minIterations = minIterations;
		this.maxIterations = maxIterations;
		this.nonConvergenceCutOff = nonConvergenceCutOff;
	}

	public List<Cluster> cluster(List<FeatureSample> samples, Random rng) {
		List<Cluster> clusters = new ArrayList<>();

		//Add initial 5 clusters
		for(int i = 0; i < 5; i++) {
			Cluster cluster = new Cluster();
			cluster.setCenter(samples.get(rng.nextInt(samples.size())).position); //TODO Are duplicate/nearby centers a problem?
			clusters.add(cluster);
		}

		int I = this.maxIterations;

		boolean convergent = false;

		boolean skipToSplitClusters = false;

		for(int x = 0; x < Math.min(this.nonConvergenceCutOff, I); x++) {
			if(!skipToSplitClusters) {
				//Partition samples into clusters
				this.partitionIntoClusters(clusters, samples);

				//Remove clusters with less samples than minSamplesPerCluster and reassign orphaned samples
				this.removeSmallClusters(clusters);
			}

			//Update cluster centers and mean distances
			float clusterMeanDistance = this.updateClusters(clusters);

			if(convergent && x >= this.minIterations) {
				return clusters;
			}

			if(x % 2 == 0 || skipToSplitClusters) {
				skipToSplitClusters = false;

				int splits = this.splitClusters(clusters, clusterMeanDistance);
				if(splits > 0) {
					I += splits;
					convergent = false;

					//Return to step 2
					continue;
				} else {
					if(convergent) {
						//Return to step 2
						continue;
					}

					//Continue with step 11
					convergent = true;
				}
			}

			//Step 11
			int mergers = this.mergeClusters(clusters);
			if(mergers > 0) {
				I += mergers;
				convergent = false;

				//Return to step 2
				continue;
			} else {
				if(convergent) {
					//Return to step 2
					continue;
				}

				convergent = true;

				skipToSplitClusters = true;
			}
		}

		return clusters;
	}

	private int mergeClusters(List<Cluster> clusters) {
		class Merger {
			Cluster c1, c2;
			float distance;
		}

		List<Merger> mergers = new ArrayList<>();

		for(int i = 0; i < clusters.size(); i++) {
			Cluster c1 = clusters.get(i);

			for(int j = i + 1; j < clusters.size(); j++) {
				Cluster c2 = clusters.get(j);

				float distance = c1.center.sub(c2.center, new Vector3f()).length();

				if(distance < this.mergeDistance) {
					Merger pair = new Merger();
					pair.c1 = c1;
					pair.c2 = c2;
					pair.distance = distance;
					mergers.add(pair);
				}
			}
		}

		Collections.sort(mergers, (m1, m2) -> Float.compare(m1.distance, m2.distance));

		int merged = 0;

		for(Merger merger : mergers) {
			int Nc1 = merger.c1.samples.size();
			int Nc2 = merger.c2.samples.size();

			//Check if clusters are empty, i.e. have already been merged with another cluster
			if(Nc1 > 0 && Nc2 > 0) {
				//Add new merged cluster
				Cluster mergedCluster = new Cluster();
				clusters.add(mergedCluster);

				//Merge samples into new cluster
				List<FeatureSample> mergedSamples = new ArrayList<>();
				mergedSamples.addAll(merger.c1.samples);
				mergedSamples.addAll(merger.c2.samples);
				for(FeatureSample sample : mergedSamples) {
					this.setSampleCluster(sample, mergedCluster);
				}

				//Calculate new merged cluster center
				Vector3f mergedCenter = new Vector3f();

				mergedCenter.x += Nc1 * merger.c1.center.x;
				mergedCenter.y += Nc1 * merger.c1.center.y;
				mergedCenter.z += Nc1 * merger.c1.center.z;

				mergedCenter.x += Nc2 * merger.c2.center.x;
				mergedCenter.y += Nc2 * merger.c2.center.y;
				mergedCenter.z += Nc2 * merger.c2.center.z;

				float normalizer = 1.0f / (Nc1 + Nc2);

				mergedCenter.x *= normalizer;
				mergedCenter.y *= normalizer;
				mergedCenter.z *= normalizer;

				mergedCluster.setCenter(mergedCenter);

				//Remove original clusters
				merger.c1.clear();
				merger.c2.clear();
				clusters.remove(merger.c1);
				clusters.remove(merger.c2);

				merged++;
			}
		}

		return merged;
	}

	private int splitClusters(List<Cluster> clusters, float clusterMeanDistance) {
		List<Vector3f> standardDeviations = this.computeClusterStandardDeviations(clusters);

		class Split {
			Cluster cluster;
			int maxComponentDim;
			float maxComponent;
		}

		List<Split> splits = new ArrayList<>();

		for(int clusterIndex = 0; clusterIndex < clusters.size(); clusterIndex++) {
			Cluster cluster = clusters.get(clusterIndex);
			Vector3f standardDeviation = standardDeviations.get(clusterIndex);

			//Find maximum standard deviation component
			float maxComponent = 0.0f;
			int maxComponentDim = 0;
			for(int dim = 0; dim < 3; dim++) {
				float component = 0.0f;

				switch(dim) {
				case 0:
					component = standardDeviation.x;
					break;
				case 1:
					component = standardDeviation.y;
					break;
				case 2:
					component = standardDeviation.z;
					break;
				}

				if(component > maxComponent) {
					maxComponent = component;
					maxComponentDim = dim;
				}
			}

			//Check if requirements for a split are met
			if(maxComponent > this.standardDeviation && cluster.meanDistance > clusterMeanDistance && cluster.samples.size() > 2 * (this.minSamplesPerCluster + 1)) {
				Split split = new Split();
				split.cluster = cluster;
				split.maxComponent = maxComponent;
				split.maxComponentDim = maxComponentDim;
				splits.add(split);
			}
		}

		for(Split split : splits) {
			//Remove original cluster and orphan all of its samples, they'll be reassigned immediately after
			split.cluster.clear();
			clusters.remove(split.cluster);

			Cluster c1 = new Cluster();
			Cluster c2 = new Cluster();

			float k = 0.5f;

			Vector3f offset = new Vector3f();
			switch(split.maxComponentDim) {
			case 0:
				offset.x = k * split.maxComponent;
				break;
			case 1:
				offset.y = k * split.maxComponent;
				break;
			case 2:
				offset.z = k * split.maxComponent;
				break;
			}

			//Offset centers of new clusters away from each other along max standard deviation component
			c1.setCenter(split.cluster.center.add(offset, new Vector3f()));
			c2.setCenter(split.cluster.center.sub(offset, new Vector3f()));

			clusters.add(c1);
			clusters.add(c2);
		}

		return splits.size();
	}

	private List<Vector3f> computeClusterStandardDeviations(List<Cluster> clusters) {
		List<Vector3f> standardDeviations = new ArrayList<>(clusters.size());

		for(Cluster cluster : clusters) {
			Vector3f clusterStandardDeviations = new Vector3f();

			for(int dim = 0; dim < 3; dim++) {
				for(FeatureSample sample : cluster.samples) {
					switch(dim) {
					case 0:
						clusterStandardDeviations.x += Math.pow(sample.position.x - cluster.center.x, 2);
						break;
					case 1:
						clusterStandardDeviations.y += Math.pow(sample.position.y - cluster.center.y, 2);
						break;
					case 2:
						clusterStandardDeviations.z += Math.pow(sample.position.z - cluster.center.z, 2);
						break;
					}
				}
			}

			float normalizer = 1.0f / cluster.samples.size();

			clusterStandardDeviations.x = (float)Math.sqrt(normalizer * clusterStandardDeviations.x);
			clusterStandardDeviations.y = (float)Math.sqrt(normalizer * clusterStandardDeviations.y);
			clusterStandardDeviations.z = (float)Math.sqrt(normalizer * clusterStandardDeviations.z);

			standardDeviations.add(clusterStandardDeviations);
		}

		return standardDeviations;
	}

	private float updateClusters(List<Cluster> clusters) {
		int numSamples = 0;
		float clusterMeanDistance = 0.0f;

		for(Cluster cluster : clusters) {
			Vector3f center = new Vector3f();

			for(FeatureSample sample : cluster.samples) {
				center.x += sample.position.x;
				center.y += sample.position.y;
				center.z += sample.position.z;
			}

			float normalizer = 1.0f / cluster.samples.size();

			cluster.center.x = center.x * normalizer;
			cluster.center.y = center.y * normalizer;
			cluster.center.z = center.z * normalizer;

			float sampleMeanDistance = 0.0f;

			for(FeatureSample sample : cluster.samples) {
				float dx = sample.position.x - cluster.center.x;
				float dy = sample.position.y - cluster.center.y;
				float dz = sample.position.z - cluster.center.z;
				sampleMeanDistance += Math.sqrt(dx * dx + dy * dy + dz * dz);
			}

			cluster.meanDistance = sampleMeanDistance * normalizer;

			clusterMeanDistance += cluster.samples.size() * cluster.meanDistance;
			numSamples += cluster.samples.size();
		}

		return clusterMeanDistance / numSamples;
	}

	private void removeSmallClusters(List<Cluster> clusters) {
		List<FeatureSample> orphanedSamples = new ArrayList<>();
		Iterator<Cluster> clusterIt = clusters.iterator();
		while(clusterIt.hasNext()) {
			Cluster cluster = clusterIt.next();

			if(cluster.samples.size() < this.minSamplesPerCluster) {
				orphanedSamples.addAll(cluster.samples);
				cluster.clear();

				clusterIt.remove();
			}
		}

		this.partitionIntoClusters(clusters, orphanedSamples);
	}

	private void partitionIntoClusters(List<Cluster> clusters, List<FeatureSample> samples) {
		for(FeatureSample sample : samples) {
			Cluster closest = this.findClosestCluster(clusters, sample.position);
			this.setSampleCluster(sample, closest);
		}
	}

	private void setSampleCluster(FeatureSample sample, Cluster cluster) {
		Cluster currentCluster = sample.getCluster();

		if(currentCluster != cluster) {
			sample.setCluster(cluster);

			if(currentCluster != null) {
				currentCluster.samples.remove(sample);
			}

			if(cluster != null) {
				cluster.samples.add(sample);
			}
		}
	}

	private Cluster findClosestCluster(List<Cluster> clusters, Vector3f sample) {
		Cluster closest = null;
		float closestSqDst = Float.MAX_VALUE;

		for(Cluster cluster : clusters) {
			float sqDst = sample.sub(cluster.center, new Vector3f()).lengthSquared();

			if(sqDst < closestSqDst) {
				closest = cluster;
				closestSqDst = sqDst;
			}
		}

		return closest;
	}
}
