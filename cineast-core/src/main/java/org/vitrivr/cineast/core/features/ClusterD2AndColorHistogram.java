package org.vitrivr.cineast.core.features;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.joml.Vector2f;
import org.joml.Vector3f;
import org.vitrivr.cineast.core.config.ReadableQueryConfig;
import org.vitrivr.cineast.core.data.CorrespondenceFunction;
import org.vitrivr.cineast.core.data.FloatVectorImpl;
import org.vitrivr.cineast.core.data.distance.SegmentDistanceElement;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh.Face;
import org.vitrivr.cineast.core.data.m3d.ReadableMesh;
import org.vitrivr.cineast.core.data.score.ScoreElement;
import org.vitrivr.cineast.core.data.segments.SegmentContainer;
import org.vitrivr.cineast.core.features.abstracts.StagedFeatureModule;
import org.vitrivr.cineast.core.util.MathHelper;
import org.vitrivr.cineast.core.util.mesh.DifferenceOfLaplacian;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil.ColorCode;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil.FaceList;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil.SampleMapper;
import org.vitrivr.cineast.core.util.mesh.IsodataClustering;
import org.vitrivr.cineast.core.util.mesh.IsodataClustering.Cluster;
import org.vitrivr.cineast.core.util.mesh.TextureColorMapper;

public class ClusterD2AndColorHistogram extends StagedFeatureModule {
	public static class FeatureSample {
		public final Vector3f position;
		public final float saliency;
		public final ColorCode color;

		private Cluster cluster;

		public FeatureSample(Vector3f position, float saliency, ColorCode color) {
			this.position = position;
			this.saliency = saliency;
			this.color = color;
		}

		public void setCluster(Cluster cluster) {
			this.cluster = cluster;
		}

		public Cluster getCluster() {
			return this.cluster;
		}
	}

	/**
	 * The number of histogram bins
	 */
	private final int bins;

	/**
	 * Diffusion/smoothing strength. At value 1.0 vertex is moved to the mean position of all its neighbors.
	 * ceil(lambda) = iteration count.
	 */
	private final float lambda;

	public ClusterD2AndColorHistogram(int bins, float lambda) {
		super("cluster_d2_and_color_histogram", 20.0f /*TODO What should this be??*/, bins);
		this.bins = bins;
		this.lambda = lambda;
	}

	public ClusterD2AndColorHistogram() {
		this(20, 8.0f);
	}

	/**
	 *
	 * @param shot
	 */
	@Override
	public void processSegment(SegmentContainer shot) {
		/* Get the Mesh. */
		ReadableMesh mesh = shot.getMesh();
		if (mesh == null || mesh.isEmpty()) {
			return;
		}

		/* Extract feature and persist it. */
		float[] feature = this.featureVectorFromMesh(mesh);
		this.persist(shot.getId(), new FloatVectorImpl(feature));
	}

	/**
	 * This method represents the first step that's executed when processing query. The associated SegmentContainer is
	 * examined and feature-vectors are being generated. The generated vectors are returned by this method together with an
	 * optional weight-vector.
	 *
	 * <strong>Important: </strong> The weight-vector must have the same size as the feature-vectors returned by the method.
	 *
	 * @param sc SegmentContainer that was submitted to the feature module.
	 * @param qc A QueryConfig object that contains query-related configuration parameters. Can still be edited.
	 * @return List of feature vectors for lookup.
	 */
	@Override
	protected List<float[]> preprocessQuery(SegmentContainer sc, ReadableQueryConfig qc) {
		/* Initialize list of features. */
		List<float[]> features = new ArrayList<>();

		/* Get the Mesh. */
		ReadableMesh mesh = sc.getMesh();
		if (mesh == null || mesh.isEmpty()) {
			return features;
		}

		/* Extract feature and persist it. */
		features.add(this.featureVectorFromMesh(mesh));
		return features;
	}

	/**
	 * This method represents the last step that's executed when processing a query. A list of partial-results (DistanceElements) returned by
	 * the lookup stage is processed based on some internal method and finally converted to a list of ScoreElements. The filtered list of
	 * ScoreElements is returned by the feature module during retrieval.
	 *
	 * @param partialResults List of partial results returned by the lookup stage.
	 * @param qc A ReadableQueryConfig object that contains query-related configuration parameters.
	 * @return List of final results. Is supposed to be de-duplicated and the number of items should not exceed the number of items per module.
	 */
	@Override
	protected List<ScoreElement> postprocessQuery(List<SegmentDistanceElement> partialResults, ReadableQueryConfig qc) {
		final CorrespondenceFunction correspondence = qc.getCorrespondenceFunction().orElse(this.correspondence);
		return ScoreElement.filterMaximumScores(partialResults.stream().map(v -> v.toScore(correspondence)));
	}

	private float[] featureVectorFromMesh(ReadableMesh mesh) {
		//TODO Expose all magic values

		IndexedTriMesh[] models = new IndexedTriMesh[3];
		float[][] scores = new float[3][];
		FaceList[] faceMap = null;

		for(int i = 0; i < 3; i++) {
			long time = System.currentTimeMillis();

			IndexedTriMesh model = new IndexedTriMesh(mesh);

			if(i == 0) {
				faceMap = IndexedTriMeshUtil.mapFaces(model);
			}

			int subdivs = (int)Math.ceil(this.lambda);

			//TODO Reuse vertices and scores from previous smoothing stage

			//Apply laplacian smoothing to vertices
			for(int j = 0; j < i * subdivs; j++) {
				IndexedTriMeshUtil.applyVertexDiffusion(model, faceMap, this.lambda / subdivs);
			}

			//Calculate mean curvatures
			int numVertices = model.getVertices().size();
			float[] modelScores = new float[numVertices];
			for(int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
				Vector2f principalCurvatures = IndexedTriMeshUtil.principalCurvatures(model, faceMap, vertexIndex);
				modelScores[vertexIndex] = 0.5f * (principalCurvatures.x + principalCurvatures.y);
			}

			//Apply laplacian smoothing to curvatures
			for(int j = 0; j < i * subdivs; j++) {
				IndexedTriMeshUtil.applyScalarDiffusion(model, faceMap, this.lambda / subdivs, modelScores);
			}

			//Scale model to have a surface area of 100
			IndexedTriMeshUtil.scaleSurface(model, 100.0f);

			System.out.println("Loaded model with " + i + " smoothing steps in " + (System.currentTimeMillis() - time) + "ms");

			models[i] = model;
			scores[i] = modelScores;
		}

		BufferedImage image = mesh.getTexture();

		SampleMapper<Float> saliencyMapper = new DifferenceOfLaplacian(scores);

		SampleMapper<ColorCode> colorMapper;
		if(image != null) {
			colorMapper = new TextureColorMapper(models[0], image);

			System.out.println("Found texture: " + mesh.getTexturePath());
		} else {
			colorMapper = new SampleMapper<ColorCode>() {
				@Override
				public ColorCode map(Face face, float u, float v, Vector3f position) {
					return new ColorCode(Color.WHITE, 0);
				}
			};
		}

		float totalCurvature = IndexedTriMeshUtil.totalAbsoluteGaussianCurvature(models[0], faceMap);

		System.out.println("Total curvature: " + totalCurvature);

		float a = 450.0f;
		float b = 2.0f;
		float c = 0.1f;

		int numSamples = (int)Math.max(2000 * Math.pow(Math.max((totalCurvature - a) / b, 0), c), 2000);

		System.out.println("Number of samples: " + numSamples);

		List<FeatureSample> samples = IndexedTriMeshUtil.sampleFeatures(models[0], saliencyMapper, colorMapper, new ArrayList<>(), numSamples, new Random());

		//Collect saliency samples
		Collections.sort(samples, (s1, s2) -> -Float.compare(s1.saliency, s2.saliency));
		List<FeatureSample> saliencySamples = new ArrayList<>(samples.subList(0, numSamples / 10));

		float meanEdgeLength = IndexedTriMeshUtil.meanEdgeLength(models[0]);

		//Collect color samples
		samples = IndexedTriMeshUtil.sampleFeatures(models[0], saliencyMapper, colorMapper, new ArrayList<>(), numSamples * 2, new Random());
		Collections.shuffle(samples);
		List<FeatureSample> colorSamples = IndexedTriMeshUtil.selectColorSamples(models[0], samples, 6, meanEdgeLength * 2.0f, meanEdgeLength * 1.5f, 0.8f);

		List<FeatureSample> clusterSamples = new ArrayList<>(saliencySamples.size() + colorSamples.size());
		clusterSamples.addAll(saliencySamples);
		clusterSamples.addAll(colorSamples);

		System.out.println("Number of clustered samples: " + clusterSamples.size());

		//Cluster samples
		IsodataClustering clustering = new IsodataClustering(clusterSamples.size() / 20, meanEdgeLength * 2.0f, meanEdgeLength * 10.0f, 10, 100, 200);
		List<Cluster> clusters = clustering.cluster(clusterSamples, new Random());

		//Compute ClusterAngle + Color feature
		int[] histogram = IndexedTriMeshUtil.computeCluster2DColorHistogram(clusters, this.bins);

		System.out.println("Histogram: " + Arrays.toString(histogram));

		float[] feature = new float[this.bins];
		for(int i = 0; i < this.bins; i++) {
			feature[i] = histogram[i];
		}

		/* Returns the normalized vector. */
		return MathHelper.normalizeL2(feature);
	}
}
