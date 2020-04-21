package org.vitrivr.cineast.core.util.mesh;

import org.joml.Vector3f;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh.Face;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil.SampleMapper;

public class DifferenceOfLaplacian implements SampleMapper<Float> {
	private final float[][] scores;

	public DifferenceOfLaplacian(float[][] scores) {
		this.scores = scores;
	}

	public static float scoreCurvatures(float meanCurvature2, float meanCurvature1, float meanCurvature0) {
		//TODO Is this right? Direct sum etc.
		return ((meanCurvature2 - meanCurvature1) * (meanCurvature1 - meanCurvature0));
	}

	@Override
	public Float map(Face face, float u, float v, Vector3f position) {
		float[] meanCurvatures = new float[3];

		for(int i = 0; i < 3; i++) {
			meanCurvatures[i] =
					this.scores[i][face.getVertices()[1]] * u +
					this.scores[i][face.getVertices()[2]] * v +
					this.scores[i][face.getVertices()[0]] * (1 - u - v);
		}

		return scoreCurvatures(meanCurvatures[2], meanCurvatures[1], meanCurvatures[0]);
	}
}