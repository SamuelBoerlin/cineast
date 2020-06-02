package org.vitrivr.cineast.core.util.mesh;

import java.awt.Color;
import java.awt.image.BufferedImage;

import javax.annotation.Nullable;

import org.joml.Matrix3f;
import org.joml.Vector2f;
import org.joml.Vector3f;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh;
import org.vitrivr.cineast.core.data.m3d.IndexedTriMesh.Face;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil.ColorCode;
import org.vitrivr.cineast.core.util.mesh.IndexedTriMeshUtil.SampleMapper;

public class TextureColorMapper implements SampleMapper<ColorCode> {
	private static final Matrix3f CIE_XYZ_TRANSFORM = new Matrix3f();

	private static final float DELTA1 = 6.0f / 29.0f;
	private static final float DELTA2 = DELTA1 * DELTA1;
	private static final float DELTA3 = DELTA2 * DELTA1;

	//https://en.wikipedia.org/wiki/CIELAB_color_space
	private static final float Xn = 95.0489f;
	private static final float Yn = 100.0f;
	private static final float Zn = 108.8840f;

	static {
		float scale = 1.0f / 0.17697f;
		CIE_XYZ_TRANSFORM.m00 = scale * 0.49f;
		CIE_XYZ_TRANSFORM.m10 = scale * 0.17697f;
		CIE_XYZ_TRANSFORM.m20 = scale * 0.0f;
		CIE_XYZ_TRANSFORM.m01 = scale * 0.31f;
		CIE_XYZ_TRANSFORM.m11 = scale * 0.81240f;
		CIE_XYZ_TRANSFORM.m21 = scale * 0.01f;
		CIE_XYZ_TRANSFORM.m02 = scale * 0.20f;
		CIE_XYZ_TRANSFORM.m12 = scale * 0.01063f;
		CIE_XYZ_TRANSFORM.m22 = scale * 0.99f;
	}

	private final IndexedTriMesh model;

	@Nullable
	private final BufferedImage texture;

	public TextureColorMapper(IndexedTriMesh model, @Nullable BufferedImage texture) {
		this.model = model;
		this.texture = texture;
	}

	public static Color sampleColor(IndexedTriMesh model, Face face, float u, float v, BufferedImage image) {
		Vector2f texUv0 = model.getUVs().get(face.getUVs()[0]);
		Vector2f texUv1 = model.getUVs().get(face.getUVs()[1]);
		Vector2f texUv2 = model.getUVs().get(face.getUVs()[2]);

		float texU = texUv1.x * u + texUv2.x * v + texUv0.x * (1 - u - v);
		float texV = 1 - (texUv1.y * u + texUv2.y * v + texUv0.y * (1 - u - v));

		int px = Math.max(0, Math.min(image.getWidth(), (int)Math.floor(texU * image.getWidth())));
		int py = Math.max(0, Math.min(image.getHeight(), (int)Math.floor(texV * image.getHeight())));

		//TODO Nearest neighbor or interpolate?
		int rgb = image.getRGB(px, py);

		return new Color((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, (rgb >> 24) & 0xFF);
	}

	private static float nonLinearResponse(float t) {
		if(t > DELTA3) {
			return (float)Math.pow(t, 1.0f / 3.0f);
		} else {
			return 1.0f / 3.0f / DELTA2 + 4.0f / 29.0f;
		}
	}

	public static int cieLabCode(Color color) {
		Vector3f rgb = new Vector3f(color.getRed(), color.getGreen(), color.getBlue());

		Vector3f xyz = new Vector3f(
				rgb.x * CIE_XYZ_TRANSFORM.m00 + rgb.y * CIE_XYZ_TRANSFORM.m01 + rgb.z * CIE_XYZ_TRANSFORM.m02,
				rgb.x * CIE_XYZ_TRANSFORM.m10 + rgb.y * CIE_XYZ_TRANSFORM.m11 + rgb.z * CIE_XYZ_TRANSFORM.m12,
				rgb.x * CIE_XYZ_TRANSFORM.m20 + rgb.y * CIE_XYZ_TRANSFORM.m21 + rgb.z * CIE_XYZ_TRANSFORM.m22
				);

		float l = 116.0f * nonLinearResponse(xyz.y / Yn) - 16.0f;
		float a = 500.0f * (nonLinearResponse(xyz.x / Xn) - nonLinearResponse(xyz.y / Yn));
		float b = 200.0f * (nonLinearResponse(xyz.y / Yn) - nonLinearResponse(xyz.z / Zn));

		int code = 0;

		if(l < 0) {
			code += 80;
		} else if(l < 15) {
			code += 0;
		} else if(l < 30) {
			code += 16;
		} else if(l < 45) {
			code += 32;
		} else if(l < 60) {
			code += 48;
		} else if(l < 75) {
			code += 64;
		} else {
			code += 80;
		}

		if(a < -100) {
			code += 12;
		} else if(a < -50) {
			code += 0;
		} else if(a < 0) {
			code += 4;
		} else if(a < 50) {
			code += 8;
		} else {
			code += 12;
		}

		if(b < -100) {
			code += 3;
		} else if(b < -50) {
			code += 0;
		} else if(b < 0) {
			code += 1;
		} else if(b < 50) {
			code += 2;
		} else {
			code += 3;
		}

		return code;
	}

	@Override
	public ColorCode map(Face face, float u, float v, Vector3f position, Color color) {
		if(this.texture != null) {
			//Multiply texture and vertex color together
			Color textureColor = sampleColor(this.model, face, u, v, this.texture);
			Color combinedColor = new Color(
					textureColor.getRed() / 255.0f * color.getRed() / 255.0f,
					textureColor.getGreen() / 255.0f * color.getGreen() / 255.0f,
					textureColor.getBlue() / 255.0f * color.getBlue() / 255.0f
					);
			return new ColorCode(combinedColor, cieLabCode(combinedColor));
		} else {
			//No texture, use vertex color
			return new ColorCode(color, cieLabCode(color));
		}
	}
}
