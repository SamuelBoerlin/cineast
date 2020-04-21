package org.vitrivr.cineast.core.extraction.segmenter.model;

import org.vitrivr.cineast.core.data.entities.MediaObjectMetadataDescriptor;
import org.vitrivr.cineast.core.data.entities.MediaSegmentDescriptor;
import org.vitrivr.cineast.core.data.m3d.Mesh;
import org.vitrivr.cineast.core.data.segments.Model3DSegment;
import org.vitrivr.cineast.core.data.segments.SegmentContainer;
import org.vitrivr.cineast.core.db.dao.writer.MediaObjectMetadataWriter;
import org.vitrivr.cineast.core.extraction.ExtractionContextProvider;
import org.vitrivr.cineast.core.extraction.segmenter.general.PassthroughSegmenter;

public class ModelSegmenter extends PassthroughSegmenter<Mesh> {
	public static final String TEXTURE_METADATA_DOMAIN = "TEXTURE";
	public static final String TEXTURE_METADATA_KEY = "texture_path";

	private final MediaObjectMetadataWriter metadataWriter;

	public ModelSegmenter(ExtractionContextProvider context) {
		this.metadataWriter = new MediaObjectMetadataWriter(context.persistencyWriter().get(), context.getBatchsize());
	}

	@Override
	protected SegmentContainer getSegmentFromContent(Mesh content) {
		return new Model3DSegment(content);
	}

	@Override
	public void persist(SegmentContainer container, MediaSegmentDescriptor descriptor) {
		String texturePath = ((Model3DSegment) container).getMesh().getTexturePath();
		if(texturePath != null) {
			this.metadataWriter.write(MediaObjectMetadataDescriptor.of(descriptor.getObjectId(), TEXTURE_METADATA_DOMAIN, TEXTURE_METADATA_KEY, texturePath));
			this.metadataWriter.flush();
		}
	}

	@Override
	public synchronized void close() {
		try {
			super.close();
		} finally {
			this.metadataWriter.close();
		}
	}
}
