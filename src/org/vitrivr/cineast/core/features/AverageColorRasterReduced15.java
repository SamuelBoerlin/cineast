package org.vitrivr.cineast.core.features;

import java.util.function.Supplier;

import org.vitrivr.cineast.core.data.MultiImage;
import org.vitrivr.cineast.core.data.segments.SegmentContainer;
import org.vitrivr.cineast.core.db.PersistencyWriterSupplier;
import org.vitrivr.cineast.core.setup.AttributeDefinition;
import org.vitrivr.cineast.core.setup.AttributeDefinition.AttributeType;
import org.vitrivr.cineast.core.setup.EntityCreator;
import org.vitrivr.cineast.core.util.ColorReductionUtil;

public class AverageColorRasterReduced15 extends AverageColorRaster {

	@Override
	public void init(PersistencyWriterSupplier supply) {
		this.phandler = supply.get();
		this.phandler.open("features_AverageColorRasterReduced15");
		this.phandler.setFieldNames("id", "raster", "hist");
	}

	@Override
	MultiImage getMultiImage(SegmentContainer shot){
		return ColorReductionUtil.quantize15(shot.getAvgImg());
	}
	
	@Override
	public void initalizePersistentLayer(Supplier<EntityCreator> supply) {
		supply.get().createFeatureEntity("features_AverageColorRasterReduced15", true, new AttributeDefinition("hist", AttributeType.VECTOR, 15), new AttributeDefinition("raster", AttributeType.VECTOR, 64));
	}

	@Override
	public void dropPersistentLayer(Supplier<EntityCreator> supply) {
		supply.get().dropEntity("features_AverageColorRasterReduced15");
	}
}
