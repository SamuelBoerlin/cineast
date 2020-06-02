package org.vitrivr.cineast.core.util.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import gnu.trove.map.hash.TObjectIntHashMap;

import org.joml.Vector2f;
import org.joml.Vector3f;
import org.joml.Vector3i;
import org.vitrivr.cineast.core.data.m3d.Mesh;
import org.vitrivr.cineast.core.util.LogHelper;

import java.io.IOException;

public class ColorMeshParser extends DataURLParser {

	/** Some format specific constants. */
	private static final String MIME_TYPE = "application/3d-color-json";

	public static Mesh parseColorGeometry(String dataUrl) {
		/* Convert Base64 string into byte array. */
		byte[] bytes = dataURLtoByteArray(dataUrl, MIME_TYPE);

		ObjectMapper mapper = new ObjectMapper();
		try {
			/* Read the JSON structure of the transmitted mesh data. */
			JsonNode node = mapper.readTree(bytes);
			JsonNode vertices = node.get("vertices");
			if (vertices == null || !vertices.isArray() || vertices.size() == 0)  {
				LOGGER.error("Submitted mesh does not contain any vertices. Aborting...");
				return Mesh.EMPTY;
			}

			/* Create new Mesh. */
			Mesh mesh = new Mesh(vertices.size() / 18, vertices.size() / 6);

			/* Prepare helper structures. */
			TObjectIntHashMap<Mesh.Vertex> vertexBuffer = new TObjectIntHashMap<>();
			int vertexIndex = 0;
			int[] vertexindices = new int[3];

			/* Add all the vertices and normals in the structure. */
			for (int i = 0; i <= vertices.size() - 18; i += 18) {
				for (int j = 0; j < 3; j++) {
					int idx = i + 6 * j;

					Vector3f color = new Vector3f((float)vertices.get(idx).asDouble(), (float)vertices.get(idx + 1).asDouble(), (float)vertices.get(idx + 2).asDouble());
					Vector3f position = new Vector3f((float)vertices.get(idx + 3).asDouble(), (float)vertices.get(idx + 4).asDouble(), (float)vertices.get(idx + 5).asDouble());

					Mesh.Vertex vertex = mesh.new Vertex(position, color, new Vector3f(0.0f, 0.0f, 0.0f), new Vector2f(0.0f, 0.0f));

					if (!vertexBuffer.containsKey(vertex)) {
						vertexBuffer.put(vertex, vertexIndex++);
						mesh.addVertex(position, color, new Vector3f(0.0f, 0.0f, 0.0f), new Vector2f(0.0f, 0.0f));
					}

					vertexindices[j] = vertexBuffer.get(vertex);
				}

				mesh.addFace(new Vector3i(vertexindices[0], vertexindices[1], vertexindices[2]));
			}

			return mesh;
		} catch (IOException e) {
			LOGGER.error("Could not create 3d color mesh from Base64 input because the file-format is not supported. {}", LogHelper.getStackTrace(e));
			return Mesh.EMPTY;
		}
	}

	public static boolean isValidColorGeometry(String dataUrl) {
		return isValidDataUrl(dataUrl, MIME_TYPE);
	}
}
