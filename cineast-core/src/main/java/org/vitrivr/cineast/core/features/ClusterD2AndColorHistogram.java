package org.vitrivr.cineast.core.features;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.joml.Vector2f;
import org.joml.Vector3f;
import org.vitrivr.cineast.core.config.QueryConfig;
import org.vitrivr.cineast.core.config.ReadableQueryConfig;
import org.vitrivr.cineast.core.data.CorrespondenceFunction;
import org.vitrivr.cineast.core.data.FloatVectorImpl;
import org.vitrivr.cineast.core.data.distance.SegmentDistanceElement;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh.Face;
import org.vitrivr.cineast.core.data.m3d.ReadableMesh;
import org.vitrivr.cineast.core.data.score.ScoreElement;
import org.vitrivr.cineast.core.data.score.SegmentScoreElement;
import org.vitrivr.cineast.core.data.segments.SegmentContainer;
import org.vitrivr.cineast.core.features.abstracts.StagedFeatureModule;
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

	/**
	 * How many iterations to calculate the histogram 
	 * average.
	 */
	private final int averageIterations;

	/**
	 * Whether the Jensen-Shannon divergence should be used to post-process the query and calculate the 
	 * similarity scores.
	 */
	private boolean useJensenShannonDivergence;

	private boolean debug = true;

	public ClusterD2AndColorHistogram(int bins, float lambda, int averageIterations, boolean useJensenShannonDivergence) {
		super("cluster_d2_and_color_histogram", (float)Math.sqrt(4 * bins) /*TODO Not sure if this is correct. For normalized histograms the max distance for each component is 2, hence sqrt(4*N) is max possible distance?*/, bins);
		this.bins = bins;
		this.lambda = lambda;
		this.averageIterations = averageIterations;
		this.useJensenShannonDivergence = useJensenShannonDivergence;
	}

	public ClusterD2AndColorHistogram() {
		this(20, 8.0f, 3, true);
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

	@Override
	protected QueryConfig defaultQueryConfig(ReadableQueryConfig qc) {
		QueryConfig defaultConf = new QueryConfig(qc)
				.setCorrespondenceFunctionIfEmpty(this.correspondence)
				.setDistanceIfEmpty(QueryConfig.Distance.chisquared);
		if(this.debug) {
			defaultConf.setMaxResults(2000).setResultsPerModule(2000);
		}
		return defaultConf;
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
	protected List<ScoreElement> postprocessQuery(List<float[]> features, List<SegmentDistanceElement> partialResults, ReadableQueryConfig qc) {
		//Unary correspondence mapping
		/*final CorrespondenceFunction correspondence = qc.getCorrespondenceFunction().orElse(this.correspondence);
		return ScoreElement.filterMaximumScores(partialResults.stream().map(v -> v.toScore(correspondence)));*/

		if(this.useJensenShannonDivergence) {
			//Jensen-Shannon Divergence as similarity measure
			float[] feature = features.get(0);
			return ScoreElement.filterMaximumScores(partialResults.stream().map(result -> {
				List<float[]> resultFeaturesList = this.selector.getFeatureVectors("id", result.getSegmentId(), "feature");
				if (resultFeaturesList.isEmpty()) {
					LOGGER.warn("No features could be fetched for the provided segmentId '{}'. Skipping post processing...", result.getSegmentId());
					return null;
				}

				float[] resultFeature = resultFeaturesList.get(0);

				return (ScoreElement) new SegmentScoreElement(result.getSegmentId(), Math.min(1.0f, Math.max(0.0f, 0.99f - this.calculateJensenShannonDivergence(feature, resultFeature))));
			}).filter(score -> score != null));
		} else {
			//Use regular correspondence to calculate score.
			//By default a linear correspondence function that scales the distance
			//to 0-1.
			final CorrespondenceFunction correspondence = qc.getCorrespondenceFunction().orElse(this.correspondence);
			return ScoreElement.filterMaximumScores(partialResults.stream().map(v -> v.toScore(correspondence)));
		}
	}

	/**
	 * Calculates a feature vector from the specified mesh using the ClusterD2/Angle+Color algorithm
	 * @param mesh Mesh to extract feature vector of
	 * @return Feature vector of mesh
	 */
	private float[] featureVectorFromMesh(ReadableMesh mesh) {
		//TODO Expose all magic values

		IndexedTriMesh[] models = new IndexedTriMesh[4];
		float[][] scores = new float[4][];
		FaceList[] faceMap = null;

		for(int i = 0; i < 4; i++) {
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

			//Scale model to have a surface area of 100
			IndexedTriMeshUtil.scaleSurface(model, 100.0f);

			if(this.debug) {
				File file = new File("debug_model_output_" + i + ".obj");

				if(file.exists()) {
					file.delete();
				}

				try(FileWriter fw = new FileWriter(file)) {
					for(Vector3f vertex : model.getVertices()) {
						fw.write("v " + vertex.x + " " + vertex.y + " " + vertex.z + "\n");
					}
					for(Face face : model.getFaces()) {
						fw.write("f " + (face.getVertices()[0] + 1) + " " + (face.getVertices()[1] + 1) + " " + (face.getVertices()[2] + 1) + "\n");
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
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

			LOGGER.info("Loaded model with {} smoothing steps in {}ms", i, (System.currentTimeMillis() - time));

			models[i] = model;
			scores[i] = modelScores;
		}

		BufferedImage image = mesh.getTexture();

		SampleMapper<Float> saliencyMapper = new DifferenceOfLaplacian(scores);

		if(image != null) {
			LOGGER.info("Found model texture: {}", mesh.getTexturePath());
		}

		SampleMapper<ColorCode> colorMapper = new TextureColorMapper(models[0], image);

		float totalCurvature = IndexedTriMeshUtil.totalAbsoluteGaussianCurvature(models[0], faceMap);

		LOGGER.info("Total curvature: {}", totalCurvature);

		float a = 450.0f;
		float b = 2.0f;
		float c = 0.1f;

		int numSamples = (int)Math.max(2000 * Math.pow(Math.max((totalCurvature - a) / b, 0), c), 2000);

		LOGGER.info("Number of samples: {}", numSamples);

		int[] totalHistogram = new int[this.bins];

		for(int i = 0; i < this.averageIterations; i++) {
			LOGGER.info("Sample iteration {}", i);

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

			LOGGER.info("Number of clustered samples: {}", clusterSamples.size());

			//Cluster samples
			IsodataClustering clustering = new IsodataClustering(clusterSamples.size() / 20, meanEdgeLength * 2.0f, meanEdgeLength * 10.0f, 10, 100, 200);
			List<Cluster> clusters = clustering.cluster(clusterSamples, new Random());

			//Compute ClusterAngle + Color feature
			int[] histogram = IndexedTriMeshUtil.computeCluster2DColorHistogram(clusters, this.bins);

			LOGGER.info("Histogram: {}", Arrays.toString(histogram));

			for(int j = 0; j < this.bins; j++) {
				totalHistogram[j] += histogram[j];
			}
		}

		float maxBin = 0.0f;

		float[] feature = new float[this.bins];
		for(int i = 0; i < this.bins; i++) {
			feature[i] = totalHistogram[i] / (float)this.averageIterations;
			maxBin = Math.max(maxBin, feature[i]);
		}

		LOGGER.info("Avg. Histogram: {}", Arrays.toString(feature));

		if(maxBin > 0 && !this.useJensenShannonDivergence) {
			//Normalize histogram
			for(int i = 0; i < this.bins; i++) {
				feature[i] = feature[i] / maxBin;
			}
		}

		return feature;
	}

	/**
	 * Calculates the Jensen-Shannon divergence between the two specified features.
	 * The function is symmetric, it doesn't matter which feature is the first or second.
	 * @param feature1 First feature
	 * @param feature2 Second feature
	 * @return Jensen-Shannon divergence between the two specified features.
	 */
	private float calculateJensenShannonDivergence(float[] feature1, float[] feature2) {
		float sum1 = 0;
		for(float value : feature1) {
			sum1 += value;
		}

		float sum2 = 0;
		for(float value : feature2) {
			sum2 += value;
		}

		float sum3 = 0.5f * (sum1 + sum2);
		float[] feature3 = new float[feature1.length];
		for(int i = 0; i < feature1.length; i++) {
			feature3[i] = 0.5f * (feature1[i] + feature2[i]);
		}

		PolynomialSplineFunction s1 = this.splineDeriv2(feature1, 1.0f / sum1);
		PolynomialSplineFunction s2 = this.splineDeriv2(feature2, 1.0f / sum2);
		PolynomialSplineFunction s3 = this.splineDeriv2(feature3, 1.0f / sum3);

		float bw1 = this.estimateBandwidth(feature1, s1);
		float bw2 = this.estimateBandwidth(feature2, s2);
		float bw3 = this.estimateBandwidth(feature3, s3);

		float end = 0.5f / Math.max(bw1, Math.max(bw2, bw3));
		float start = -end;
		float diff = end - start;

		int samples = (int)Math.ceil(diff * 100);

		double sum = 0;

		double dkl13 = 0.0;
		for(int i = 0; i < samples - 1; i++) {
			float x1 = i / (float)(samples - 1) * diff + start;
			float x2 = (i + 1) / (float)(samples - 1) * diff + start;

			float dx = x2 - x1;
			float mid = 0.5f * (x1 + x2);

			double px = this.estimateDensity(feature1, sum1, bw1, mid);
			double qx = this.estimateDensity(feature3, sum3, bw3, mid);

			sum += px * dx;

			dkl13 += px * divLog(px, qx, 0.0000001D) * dx;
		}

		//TODO Debug
		System.out.println("Sum: " + sum + " " + diff + " " + samples);

		double dkl23 = 0.0;
		for(int i = 0; i < samples - 1; i++) {
			float x1 = i / (float)(samples - 1) * diff + start;
			float x2 = (i + 1) / (float)(samples - 1) * diff + start;

			float dx = x2 - x1;
			float mid = 0.5f * (x1 + x2);

			double px = this.estimateDensity(feature2, sum2, bw2, mid);
			double qx = this.estimateDensity(feature3, sum3, bw3, mid);

			dkl23 += px * divLog(px, qx, 0.0000001D) * dx;
		}

		return (float)(0.5f * dkl13 + 0.5f * dkl23);
	}

	private double divLog(double numerator, double denominator, double epsilon) {
		if(Math.abs(numerator) < epsilon && Math.abs(denominator) < epsilon) {
			return 0;
		}
		return Math.log(numerator / denominator);
	}

	/**
	 * Estimates the density at a specific x for the given feature.
	 * @param feature Feature to estimate density of
	 * @param numSamples Number of samples (sum of feature)
	 * @param bandwidth Kernel bandwidth
	 * @param x Value to estimate density at
	 * @return Density estimation of the feature at value x
	 */
	private double estimateDensity(float[] feature, float numSamples, double bandwidth, float x) {
		double density = 0.0;

		for(int i = 0; i < feature.length; i++) {
			float sx = i / (float)(feature.length - 1);

			double arg = (x - sx) / (double)bandwidth;
			density += 1 / Math.sqrt(2 * Math.PI) * Math.exp(-0.5D * arg * arg) * feature[i];
		}

		return density / (numSamples * bandwidth);
	}

	/**
	 * Creates the spline of the second derivative of the given feature.
	 * @param feature Feature to take second derivative of
	 * @param scale Feature scale
	 * @return Spline of the second derivative of the given feature
	 */
	private PolynomialSplineFunction splineDeriv2(float[] feature, float scale) {
		SplineInterpolator interpolator = new SplineInterpolator();

		double[] sx = new double[feature.length];
		double[] sy = new double[feature.length];
		for(int i = 0; i < feature.length; i++) {
			sx[i] = i / (float)(feature.length - 1);
			sy[i] = feature[i] * scale;
		}

		return interpolator.interpolate(sx, sy).polynomialSplineDerivative().polynomialSplineDerivative();
	}

	/**
	 * Estimates a kernel bandwidth h with the specified second derivative ({@link #splineDeriv2(float[], float)} of
	 * the feature.
	 * @param feature Feature to estimate bandwidth for
	 * @param c Second derivative of the feature
	 * @return Kernel bandwidth estimation
	 */
	private float estimateBandwidth(float[] feature, PolynomialSplineFunction c) {
		double[] knots = c.getKnots();

		//Second derivative of cubic polynomial is linear,
		//so only the end points of the linear segments need to be sampled and the integrals
		//between can be calculated analytically
		double integralcsq = 0.0;
		for(int i = 0;  i < knots.length - 1; i++) {
			float x1 = (float)knots[i];
			float x2 = (float)knots[i + 1];
			float y1 = (float)c.value(x1);
			float y2 = (float)c.value(x2);

			//Calculate linear segment parameters
			float dx = x2 - x1;
			float a = (y2 - y1) / dx;
			float b = y1;

			//Integral of linear segment squared (a*x + b)^2
			float n1 = b;
			float n2 = a * dx + b;
			float piece = (n2 * n2 * n2 - n1 * n1 * n1) / (3.0f * a);

			//Add to total integral
			integralcsq += piece;
		}

		double rk = 0.282095; //Integral of gaussian with variance=1, mean=0, from -inf to +inf = 1 / (2 * sqrt(pi))
		double mu22 = 1.0; //Integral of c * x^2 * exp(-a*x^2), from -inf to +inf = c * sqrt(pi) / (2 * a^(3/2)). With c=1/sqrt(pi), a=1/2 --> integral = 1

		int n = feature.length;

		//Bandwidth estimation
		float h = (float)(Math.pow(rk / (mu22 * integralcsq), 1 / 5.0) * Math.pow(n, -1 / 5.0));

		return h;
	}
}
