/*******************************************************************************

  Intel(R) Gigabit Ethernet Linux driver
  Copyright(c) 2007-2009 Intel Corporation.

  This program is free software; you can redistribute it and/or modify it
  under the terms and conditions of the GNU General Public License,
  version 2, as published by the Free Software Foundation.

  This program is distributed in the hope it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.

  The full GNU General Public License is included in this distribution in
  the file called "COPYING".

  Contact Information:
  e1000-devel Mailing List <e1000-devel@lists.sourceforge.net>
  Intel Corporation, 5200 N.E. Elam Young Parkway, Hillsboro, OR 97124-6497

*******************************************************************************/

FILE_LICENCE ( GPL2_ONLY );

/*
 * 82575EB Gigabit Network Connection
 * 82575EB Gigabit Backplane Connection
 * 82575GB Gigabit Network Connection
 * 82576 Gigabit Network Connection
 * 82576 Quad Port Gigabit Mezzanine Adapter
 */

#include "igb.h"

static s32  igb_init_phy_params_82575(struct e1000_hw *hw);
static s32  igb_init_nvm_params_82575(struct e1000_hw *hw);
static s32  igb_init_mac_params_82575(struct e1000_hw *hw);
static s32  igb_acquire_phy_82575(struct e1000_hw *hw);
static void igb_release_phy_82575(struct e1000_hw *hw);
static s32  igb_acquire_nvm_82575(struct e1000_hw *hw);
static void igb_release_nvm_82575(struct e1000_hw *hw);
static s32  igb_check_for_link_82575(struct e1000_hw *hw);
static s32  igb_get_cfg_done_82575(struct e1000_hw *hw);
static s32  igb_get_link_up_info_82575(struct e1000_hw *hw, u16 *speed,
                                         u16 *duplex);
static s32  igb_init_hw_82575(struct e1000_hw *hw);
static s32  igb_phy_hw_reset_sgmii_82575(struct e1000_hw *hw);
static s32  igb_read_phy_reg_sgmii_82575(struct e1000_hw *hw, u32 offset,
                                           u16 *data);
static s32  igb_reset_hw_82575(struct e1000_hw *hw);
static s32  igb_set_d0_lplu_state_82575(struct e1000_hw *hw,
                                          bool active);
static s32  igb_setup_copper_link_82575(struct e1000_hw *hw);
static s32  igb_setup_serdes_link_82575(struct e1000_hw *hw);
static s32  igb_valid_led_default_82575(struct e1000_hw *hw, u16 *data);
static s32  igb_write_phy_reg_sgmii_82575(struct e1000_hw *hw,
                                            u32 offset, u16 data);
static void igb_clear_hw_cntrs_82575(struct e1000_hw *hw);
static s32  igb_acquire_swfw_sync_82575(struct e1000_hw *hw, u16 mask);
static s32  igb_get_pcs_speed_and_duplex_82575(struct e1000_hw *hw,
                                                 u16 *speed, u16 *duplex);
static s32  igb_get_phy_id_82575(struct e1000_hw *hw);
static void igb_release_swfw_sync_82575(struct e1000_hw *hw, u16 mask);
static bool igb_sgmii_active_82575(struct e1000_hw *hw);
static s32  igb_reset_init_script_82575(struct e1000_hw *hw);
static s32  igb_read_mac_addr_82575(struct e1000_hw *hw);
static void igb_power_down_phy_copper_82575(struct e1000_hw *hw);
static void igb_shutdown_serdes_link_82575(struct e1000_hw *hw);
static s32 igb_set_pcie_completion_timeout(struct e1000_hw *hw);

/**
 *  igb_init_phy_params_82575 - Init PHY func ptrs.
 *  @hw: pointer to the HW structure
 **/
static s32 igb_init_phy_params_82575(struct e1000_hw *hw)
{
	struct e1000_phy_info *phy = &hw->phy;
	s32 ret_val = E1000_SUCCESS;

	DEBUGFUNC("igb_init_phy_params_82575");

	if (hw->phy.media_type != e1000_media_type_copper) {
		phy->type = e1000_phy_none;
		goto out;
	}

	phy->ops.power_up   = igb_power_up_phy_copper;
	phy->ops.power_down = igb_power_down_phy_copper_82575;

	phy->autoneg_mask           = AUTONEG_ADVERTISE_SPEED_DEFAULT;
	phy->reset_delay_us         = 100;

	phy->ops.acquire            = igb_acquire_phy_82575;
	phy->ops.check_reset_block  = igb_check_reset_block_generic;
	phy->ops.commit             = igb_phy_sw_reset_generic;
	phy->ops.get_cfg_done       = igb_get_cfg_done_82575;
	phy->ops.release            = igb_release_phy_82575;

	if (igb_sgmii_active_82575(hw)) {
		phy->ops.reset      = igb_phy_hw_reset_sgmii_82575;
		phy->ops.read_reg   = igb_read_phy_reg_sgmii_82575;
		phy->ops.write_reg  = igb_write_phy_reg_sgmii_82575;
	} else {
		phy->ops.reset      = igb_phy_hw_reset_generic;
		phy->ops.read_reg   = igb_read_phy_reg_igp;
		phy->ops.write_reg  = igb_write_phy_reg_igp;
	}

	/* Set phy->phy_addr and phy->id. */
	ret_val = igb_get_phy_id_82575(hw);

	/* Verify phy id and set remaining function pointers */
	switch (phy->id) {
	case M88E1111_I_PHY_ID:
		phy->type                   = e1000_phy_m88;
		phy->ops.check_polarity     = igb_check_polarity_m88;
		phy->ops.get_info           = igb_get_phy_info_m88;
#if 0
		phy->ops.get_cable_length   = igb_get_cable_length_m88;
#endif
#if 0
		phy->ops.force_speed_duplex = igb_phy_force_speed_duplex_m88;
#endif
		break;
	case IGP03E1000_E_PHY_ID:
	case IGP04E1000_E_PHY_ID:
		phy->type                   = e1000_phy_igp_3;
		phy->ops.check_polarity     = igb_check_polarity_igp;
		phy->ops.get_info           = igb_get_phy_info_igp;
#if 0
		phy->ops.get_cable_length   = igb_get_cable_length_igp_2;
#endif
#if 0
		phy->ops.force_speed_duplex = igb_phy_force_speed_duplex_igp;
#endif
		phy->ops.set_d0_lplu_state  = igb_set_d0_lplu_state_82575;
		phy->ops.set_d3_lplu_state  = igb_set_d3_lplu_state_generic;
		break;
	default:
		ret_val = -E1000_ERR_PHY;
		goto out;
	}

out:
	return ret_val;
}

/**
 *  igb_init_nvm_params_82575 - Init NVM func ptrs.
 *  @hw: pointer to the HW structure
 **/
static s32 igb_init_nvm_params_82575(struct e1000_hw *hw)
{
	struct e1000_nvm_info *nvm = &hw->nvm;
	u32 eecd = E1000_READ_REG(hw, E1000_EECD);
	u16 size;

	DEBUGFUNC("igb_init_nvm_params_82575");

	nvm->opcode_bits        = 8;
	nvm->delay_usec         = 1;
	switch (nvm->override) {
	case e1000_nvm_override_spi_large:
		nvm->page_size    = 32;
		nvm->address_bits = 16;
		break;
	case e1000_nvm_override_spi_small:
		nvm->page_size    = 8;
		nvm->address_bits = 8;
		break;
	default:
		nvm->page_size    = eecd & E1000_EECD_ADDR_BITS ? 32 : 8;
		nvm->address_bits = eecd & E1000_EECD_ADDR_BITS ? 16 : 8;
		break;
	}

	nvm->type              = e1000_nvm_eeprom_spi;

	size = (u16)((eecd & E1000_EECD_SIZE_EX_MASK) >>
	                  E1000_EECD_SIZE_EX_SHIFT);

	/*
	 * Added to a constant, "size" becomes the left-shift value
	 * for setting word_size.
	 */
	size += NVM_WORD_SIZE_BASE_SHIFT;

	/* EEPROM access above 16k is unsupported */
	if (size > 14)
		size = 14;
	nvm->word_size = 1 << size;

	/* Function Pointers */
	nvm->ops.acquire       = igb_acquire_nvm_82575;
	nvm->ops.read          = igb_read_nvm_eerd;
	nvm->ops.release       = igb_release_nvm_82575;
	nvm->ops.update        = igb_update_nvm_checksum_generic;
	nvm->ops.valid_led_default = igb_valid_led_default_82575;
	nvm->ops.validate      = igb_validate_nvm_checksum_generic;
	nvm->ops.write         = igb_write_nvm_spi;

	return E1000_SUCCESS;
}

/**
 *  igb_init_mac_params_82575 - Init MAC func ptrs.
 *  @hw: pointer to the HW structure
 **/
static s32 igb_init_mac_params_82575(struct e1000_hw *hw)
{
	struct e1000_mac_info *mac = &hw->mac;
	struct e1000_dev_spec_82575 *dev_spec = &hw->dev_spec._82575;
	u32 ctrl_ext = 0;

	DEBUGFUNC("igb_init_mac_params_82575");

	/* Set media type */
        /*
	 * The 82575 uses bits 22:23 for link mode. The mode can be changed
         * based on the EEPROM. We cannot rely upon device ID. There
         * is no distinguishable difference between fiber and internal
         * SerDes mode on the 82575. There can be an external PHY attached
         * on the SGMII interface. For this, we'll set sgmii_active to true.
         */
	hw->phy.media_type = e1000_media_type_copper;
	dev_spec->sgmii_active = false;

	ctrl_ext = E1000_READ_REG(hw, E1000_CTRL_EXT);
	switch (ctrl_ext & E1000_CTRL_EXT_LINK_MODE_MASK) {
	case E1000_CTRL_EXT_LINK_MODE_SGMII:
		dev_spec->sgmii_active = true;
		ctrl_ext |= E1000_CTRL_I2C_ENA;
		break;
	case E1000_CTRL_EXT_LINK_MODE_PCIE_SERDES:
		hw->phy.media_type = e1000_media_type_internal_serdes;
		ctrl_ext |= E1000_CTRL_I2C_ENA;
		break;
	default:
		ctrl_ext &= ~E1000_CTRL_I2C_ENA;
		break;
	}

	E1000_WRITE_REG(hw, E1000_CTRL_EXT, ctrl_ext);

	/* Set mta register count */
	mac->mta_reg_count = 128;
	/* Set uta register count */
	mac->uta_reg_count = (hw->mac.type == e1000_82575) ? 0 : 128;
	/* Set rar entry count */
	mac->rar_entry_count = E1000_RAR_ENTRIES_82575;
	if (mac->type == e1000_82576)
		mac->rar_entry_count = E1000_RAR_ENTRIES_82576;
	/* Set if part includes ASF firmware */
	mac->asf_firmware_present = true;
	/* Set if manageability features are enabled. */
	mac->arc_subsystem_valid =
	        (E1000_READ_REG(hw, E1000_FWSM) & E1000_FWSM_MODE_MASK)
	                ? true : false;

	/* Function pointers */

	/* bus type/speed/width */
	mac->ops.get_bus_info = igb_get_bus_info_pcie_generic;
	/* reset */
	mac->ops.reset_hw = igb_reset_hw_82575;
	/* hw initialization */
	mac->ops.init_hw = igb_init_hw_82575;
	/* link setup */
	mac->ops.setup_link = igb_setup_link_generic;
	/* physical interface link setup */
	mac->ops.setup_physical_interface =
	        (hw->phy.media_type == e1000_media_type_copper)
	                ? igb_setup_copper_link_82575
	                : igb_setup_serdes_link_82575;
	/* physical interface shutdown */
	mac->ops.shutdown_serdes = igb_shutdown_serdes_link_82575;
	/* check for link */
	mac->ops.check_for_link = igb_check_for_link_82575;
	/* receive address register setting */
	mac->ops.rar_set = igb_rar_set_generic;
	/* read mac address */
	mac->ops.read_mac_addr = igb_read_mac_addr_82575;
	/* multicast address update */
	mac->ops.update_mc_addr_list = igb_update_mc_addr_list_generic;
	/* writing VFTA */
	mac->ops.write_vfta = igb_write_vfta_generic;
	/* clearing VFTA */
	mac->ops.clear_vfta = igb_clear_vfta_generic;
	/* setting MTA */
#if 0
	mac->ops.mta_set = igb_mta_set_generic;
	/* ID LED init */
	mac->ops.id_led_init = igb_id_led_init_generic;
	/* blink LED */
	mac->ops.blink_led = igb_blink_led_generic;
	/* setup LED */
	mac->ops.setup_led = igb_setup_led_generic;
	/* cleanup LED */
	mac->ops.cleanup_led = igb_cleanup_led_generic;
	/* turn on/off LED */
	mac->ops.led_on = igb_led_on_generic;
	mac->ops.led_off = igb_led_off_generic;
#endif
	/* clear hardware counters */
	mac->ops.clear_hw_cntrs = igb_clear_hw_cntrs_82575;
	/* link info */
	mac->ops.get_link_up_info = igb_get_link_up_info_82575;

	/* set lan id for port to determine which phy lock to use */
	hw->mac.ops.set_lan_id(hw);

	return E1000_SUCCESS;
}

/**
 *  igb_init_function_pointers_82575 - Init func ptrs.
 *  @hw: pointer to the HW structure
 *
 *  Called to initialize all function pointers and parameters.
 **/
void igb_init_function_pointers_82575(struct e1000_hw *hw)
{
	DEBUGFUNC("igb_init_function_pointers_82575");

	hw->mac.ops.init_params = igb_init_mac_params_82575;
	hw->nvm.ops.init_params = igb_init_nvm_params_82575;
	hw->phy.ops.init_params = igb_init_phy_params_82575;
#if 0
	hw->mbx.ops.init_params = igb_init_mbx_params_pf;
#endif
}

/**
 *  igb_acquire_phy_82575 - Acquire rights to access PHY
 *  @hw: pointer to the HW structure
 *
 *  Acquire access rights to the correct PHY.
 **/
static s32 igb_acquire_phy_82575(struct e1000_hw *hw)
{
	u16 mask = E1000_SWFW_PHY0_SM;

	DEBUGFUNC("igb_acquire_phy_82575");

	if (hw->bus.func == E1000_FUNC_1)
		mask = E1000_SWFW_PHY1_SM;

	return igb_acquire_swfw_sync_82575(hw, mask);
}

/**
 *  igb_release_phy_82575 - Release rights to access PHY
 *  @hw: pointer to the HW structure
 *
 *  A wrapper to release access rights to the correct PHY.
 **/
static void igb_release_phy_82575(struct e1000_hw *hw)
{
	u16 mask = E1000_SWFW_PHY0_SM;

	DEBUGFUNC("igb_release_phy_82575");

	if (hw->bus.func == E1000_FUNC_1)
		mask = E1000_SWFW_PHY1_SM;

	igb_release_swfw_sync_82575(hw, mask);
}

/**
 *  igb_read_phy_reg_sgmii_82575 - Read PHY register using sgmii
 *  @hw: pointer to the HW structure
 *  @offset: register offset to be read
 *  @data: pointer to the read data
 *
 *  Reads the PHY register at offset using the serial gigabit media independent
 *  interface and stores the retrieved information in data.
 **/
static s32 igb_read_phy_reg_sgmii_82575(struct e1000_hw *hw, u32 offset,
                                          u16 *data)
{
	s32 ret_val = -E1000_ERR_PARAM;

	DEBUGFUNC("igb_read_phy_reg_sgmii_82575");

	if (offset > E1000_MAX_SGMII_PHY_REG_ADDR) {
		DEBUGOUT1("PHY Address %u is out of range\n", offset);
		goto out;
	}

	ret_val = hw->phy.ops.acquire(hw);
	if (ret_val)
		goto out;

	ret_val = igb_read_phy_reg_i2c(hw, offset, data);

	hw->phy.ops.release(hw);

out:
	return ret_val;
}

/**
 *  igb_write_phy_reg_sgmii_82575 - Write PHY register using sgmii
 *  @hw: pointer to the HW structure
 *  @offset: register offset to write to
 *  @data: data to write at register offset
 *
 *  Writes the data to PHY register at the offset using the serial gigabit
 *  media independent interface.
 **/
static s32 igb_write_phy_reg_sgmii_82575(struct e1000_hw *hw, u32 offset,
                                           u16 data)
{
	s32 ret_val = -E1000_ERR_PARAM;

	DEBUGFUNC("igb_write_phy_reg_sgmii_82575");

	if (offset > E1000_MAX_SGMII_PHY_REG_ADDR) {
		DEBUGOUT1("PHY Address %d is out of range\n", offset);
		goto out;
	}

	ret_val = hw->phy.ops.acquire(hw);
	if (ret_val)
		goto out;

	ret_val = igb_write_phy_reg_i2c(hw, offset, data);

	hw->phy.ops.release(hw);

out:
	return ret_val;
}

/**
 *  igb_get_phy_id_82575 - Retrieve PHY addr and id
 *  @hw: pointer to the HW structure
 *
 *  Retrieves the PHY address and ID for both PHY's which do and do not use
 *  sgmi interface.
 **/
static s32 igb_get_phy_id_82575(struct e1000_hw *hw)
{
	struct e1000_phy_info *phy = &hw->phy;
	s32  ret_val = E1000_SUCCESS;
	u16 phy_id;
	u32 ctrl_ext;

	DEBUGFUNC("igb_get_phy_id_82575");

	/*
	 * For SGMII PHYs, we try the list of possible addresses until
	 * we find one that works.  For non-SGMII PHYs
	 * (e.g. integrated copper PHYs), an address of 1 should
	 * work.  The result of this function should mean phy->phy_addr
	 * and phy->id are set correctly.
	 */
	if (!igb_sgmii_active_82575(hw)) {
		phy->addr = 1;
		ret_val = igb_get_phy_id(hw);
		goto out;
	}

	/* Power on sgmii phy if it is disabled */
	ctrl_ext = E1000_READ_REG(hw, E1000_CTRL_EXT);
	E1000_WRITE_REG(hw, E1000_CTRL_EXT,
	                ctrl_ext & ~E1000_CTRL_EXT_SDP3_DATA);
	E1000_WRITE_FLUSH(hw);
	msec_delay(300);

	/*
	 * The address field in the I2CCMD register is 3 bits and 0 is invalid.
	 * Therefore, we need to test 1-7
	 */
	for (phy->addr = 1; phy->addr < 8; phy->addr++) {
		ret_val = igb_read_phy_reg_sgmii_82575(hw, PHY_ID1, &phy_id);
		if (ret_val == E1000_SUCCESS) {
			DEBUGOUT2("Vendor ID 0x%08X read at address %u\n",
			          phy_id,
			          phy->addr);
			/*
			 * At the time of this writing, The M88 part is
			 * the only supported SGMII PHY product.
			 */
			if (phy_id == M88_VENDOR)
				break;
		} else {
			DEBUGOUT1("PHY address %u was unreadable\n",
			          phy->addr);
		}
	}

	/* A valid PHY type couldn't be found. */
	if (phy->addr == 8) {
		phy->addr = 0;
		ret_val = -E1000_ERR_PHY;
	} else {
		ret_val = igb_get_phy_id(hw);
	}

	/* restore previous sfp cage power state */
	E1000_WRITE_REG(hw, E1000_CTRL_EXT, ctrl_ext);

out:
	return ret_val;
}

/**
 *  igb_phy_hw_reset_sgmii_82575 - Performs a PHY reset
 *  @hw: pointer to the HW structure
 *
 *  Resets the PHY using the serial gigabit media independent interface.
 **/
static s32 igb_phy_hw_reset_sgmii_82575(struct e1000_hw *hw)
{
	s32 ret_val = E1000_SUCCESS;

	DEBUGFUNC("igb_phy_hw_reset_sgmii_82575");

	/*
	 * This isn't a true "hard" reset, but is the only reset
	 * available to us at this time.
	 */

	DEBUGOUT("Soft resetting SGMII attached PHY...\n");

	if (!(hw->phy.ops.write_reg))
		goto out;

	/*
	 * SFP documentation requires the following to configure the SPF module
	 * to work on SGMII.  No further documentation is given.
	 */
	ret_val = hw->phy.ops.write_reg(hw, 0x1B, 0x8084);
	if (ret_val)
		goto out;

	ret_val = hw->phy.ops.commit(hw);

out:
	return ret_val;
}

/**
 *  igb_set_d0_lplu_state_82575 - Set Low Power Linkup D0 state
 *  @hw: pointer to the HW structure
 *  @active: true to enable LPLU, false to disable
 *
 *  Sets the LPLU D0 state according to the active flag.  When
 *  activating LPLU this function also disables smart speed
 *  and vice versa.  LPLU will not be activated unless the
 *  device autonegotiation advertisement meets standards of
 *  either 10 or 10/100 or 10/100/1000 at all duplexes.
 *  This is a function pointer entry point only called by
 *  PHY setup routines.
 **/
static s32 igb_set_d0_lplu_state_82575(struct e1000_hw *hw, bool active)
{
	struct e1000_phy_info *phy = &hw->phy;
	s32 ret_val = E1000_SUCCESS;
	u16 data;

	DEBUGFUNC("igb_set_d0_lplu_state_82575");

	if (!(hw->phy.ops.read_reg))
		goto out;

	ret_val = phy->ops.read_reg(hw, IGP02E1000_PHY_POWER_MGMT, &data);
	if (ret_val)
		goto out;

	if (active) {
		data |= IGP02E1000_PM_D0_LPLU;
		ret_val = phy->ops.write_reg(hw, IGP02E1000_PHY_POWER_MGMT,
		                             data);
		if (ret_val)
			goto out;

		/* When LPLU is enabled, we should disable SmartSpeed */
		ret_val = phy->ops.read_reg(hw, IGP01E1000_PHY_PORT_CONFIG,
		                            &data);
		data &= ~IGP01E1000_PSCFR_SMART_SPEED;
		ret_val = phy->ops.write_reg(hw, IGP01E1000_PHY_PORT_CONFIG,
		                             data);
		if (ret_val)
			goto out;
	} else {
		data &= ~IGP02E1000_PM_D0_LPLU;
		ret_val = phy->ops.write_reg(hw, IGP02E1000_PHY_POWER_MGMT,
		                             data);
		/*
		 * LPLU and SmartSpeed are mutually exclusive.  LPLU is used
		 * during Dx states where the power conservation is most
		 * important.  During driver activity we should enable
		 * SmartSpeed, so performance is maintained.
		 */
		if (phy->smart_speed == e1000_smart_speed_on) {
			ret_val = phy->ops.read_reg(hw,
			                            IGP01E1000_PHY_PORT_CONFIG,
			                            &data);
			if (ret_val)
				goto out;

			data |= IGP01E1000_PSCFR_SMART_SPEED;
			ret_val = phy->ops.write_reg(hw,
			                             IGP01E1000_PHY_PORT_CONFIG,
			                             data);
			if (ret_val)
				goto out;
		} else if (phy->smart_speed == e1000_smart_speed_off) {
			ret_val = phy->ops.read_reg(hw,
			                            IGP01E1000_PHY_PORT_CONFIG,
			                            &data);
			if (ret_val)
				goto out;

			data &= ~IGP01E1000_PSCFR_SMART_SPEED;
			ret_val = phy->ops.write_reg(hw,
			                             IGP01E1000_PHY_PORT_CONFIG,
			                             data);
			if (ret_val)
				goto out;
		}
	}

out:
	return ret_val;
}

/**
 *  igb_acquire_nvm_82575 - Request for access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Acquire the necessary semaphores for exclusive access to the EEPROM.
 *  Set the EEPROM access request bit and wait for EEPROM access grant bit.
 *  Return successful if access grant bit set, else clear the request for
 *  EEPROM access and return -E1000_ERR_NVM (-1).
 **/
static s32 igb_acquire_nvm_82575(struct e1000_hw *hw)
{
	s32 ret_val;

	DEBUGFUNC("igb_acquire_nvm_82575");

	ret_val = igb_acquire_swfw_sync_82575(hw, E1000_SWFW_EEP_SM);
	if (ret_val)
		goto out;

	ret_val = igb_acquire_nvm_generic(hw);

	if (ret_val)
		igb_release_swfw_sync_82575(hw, E1000_SWFW_EEP_SM);

out:
	return ret_val;
}

/**
 *  igb_release_nvm_82575 - Release exclusive access to EEPROM
 *  @hw: pointer to the HW structure
 *
 *  Stop any current commands to the EEPROM and clear the EEPROM request bit,
 *  then release the semaphores acquired.
 **/
static void igb_release_nvm_82575(struct e1000_hw *hw)
{
	DEBUGFUNC("igb_release_nvm_82575");

	igb_release_nvm_generic(hw);
	igb_release_swfw_sync_82575(hw, E1000_SWFW_EEP_SM);
}

/**
 *  igb_acquire_swfw_sync_82575 - Acquire SW/FW semaphore
 *  @hw: pointer to the HW structure
 *  @mask: specifies which semaphore to acquire
 *
 *  Acquire the SW/FW semaphore to access the PHY or NVM.  The mask
 *  will also specify which port we're acquiring the lock for.
 **/
static s32 igb_acquire_swfw_sync_82575(struct e1000_hw *hw, u16 mask)
{
	u32 swfw_sync;
	u32 swmask = mask;
	u32 fwmask = mask << 16;
	s32 ret_val = E1000_SUCCESS;
	s32 i = 0, timeout = 200; /* FIXME: find real value to use here */

	DEBUGFUNC("igb_acquire_swfw_sync_82575");

	while (i < timeout) {
		if (igb_get_hw_semaphore_generic(hw)) {
			ret_val = -E1000_ERR_SWFW_SYNC;
			goto out;
		}

		swfw_sync = E1000_READ_REG(hw, E1000_SW_FW_SYNC);
		if (!(swfw_sync & (fwmask | swmask)))
			break;

		/*
		 * Firmware currently using resource (fwmask)
		 * or other software thread using resource (swmask)
		 */
		igb_put_hw_semaphore_generic(hw);
		msec_delay_irq(5);
		i++;
	}

	if (i == timeout) {
		DEBUGOUT("Driver can't access resource, SW_FW_SYNC timeout.\n");
		ret_val = -E1000_ERR_SWFW_SYNC;
		goto out;
	}

	swfw_sync |= swmask;
	E1000_WRITE_REG(hw, E1000_SW_FW_SYNC, swfw_sync);

	igb_put_hw_semaphore_generic(hw);

out:
	return ret_val;
}

/**
 *  igb_release_swfw_sync_82575 - Release SW/FW semaphore
 *  @hw: pointer to the HW structure
 *  @mask: specifies which semaphore to acquire
 *
 *  Release the SW/FW semaphore used to access the PHY or NVM.  The mask
 *  will also specify which port we're releasing the lock for.
 **/
static void igb_release_swfw_sync_82575(struct e1000_hw *hw, u16 mask)
{
	u32 swfw_sync;

	DEBUGFUNC("igb_release_swfw_sync_82575");

	while (igb_get_hw_semaphore_generic(hw) != E1000_SUCCESS);
	/* Empty */

	swfw_sync = E1000_READ_REG(hw, E1000_SW_FW_SYNC);
	swfw_sync &= ~mask;
	E1000_WRITE_REG(hw, E1000_SW_FW_SYNC, swfw_sync);

	igb_put_hw_semaphore_generic(hw);
}

/**
 *  igb_get_cfg_done_82575 - Read config done bit
 *  @hw: pointer to the HW structure
 *
 *  Read the management control register for the config done bit for
 *  completion status.  NOTE: silicon which is EEPROM-less will fail trying
 *  to read the config done bit, so an error is *ONLY* logged and returns
 *  E1000_SUCCESS.  If we were to return with error, EEPROM-less silicon
 *  would not be able to be reset or change link.
 **/
static s32 igb_get_cfg_done_82575(struct e1000_hw *hw)
{
	s32 timeout = PHY_CFG_TIMEOUT;
	s32 ret_val = E1000_SUCCESS;
	u32 mask = E1000_NVM_CFG_DONE_PORT_0;

	DEBUGFUNC("igb_get_cfg_done_82575");

	if (hw->bus.func == E1000_FUNC_1)
		mask = E1000_NVM_CFG_DONE_PORT_1;
	while (timeout) {
		if (E1000_READ_REG(hw, E1000_EEMNGCTL) & mask)
			break;
		msec_delay(1);
		timeout--;
	}
	if (!timeout) {
		DEBUGOUT("MNG configuration cycle has not completed.\n");
        }

	/* If EEPROM is not marked present, init the PHY manually */
	if (((E1000_READ_REG(hw, E1000_EECD) & E1000_EECD_PRES) == 0) &&
	    (hw->phy.type == e1000_phy_igp_3))
		igb_phy_init_script_igp3(hw);

	return ret_val;
}

/**
 *  igb_get_link_up_info_82575 - Get link speed/duplex info
 *  @hw: pointer to the HW structure
 *  @speed: stores the current speed
 *  @duplex: stores the current duplex
 *
 *  This is a wrapper function, if using the serial gigabit media independent
 *  interface, use PCS to retrieve the link speed and duplex information.
 *  Otherwise, use the generic function to get the link speed and duplex info.
 **/
static s32 igb_get_link_up_info_82575(struct e1000_hw *hw, u16 *speed,
                                        u16 *duplex)
{
	s32 ret_val;

	DEBUGFUNC("igb_get_link_up_info_82575");

	if (hw->phy.media_type != e1000_media_type_copper)
		ret_val = igb_get_pcs_speed_and_duplex_82575(hw, speed,
		                                               duplex);
	else
		ret_val = igb_get_speed_and_duplex_copper_generic(hw, speed,
		                                                    duplex);

	return ret_val;
}

/**
 *  igb_check_for_link_82575 - Check for link
 *  @hw: pointer to the HW structure
 *
 *  If sgmii is enabled, then use the pcs register to determine link, otherwise
 *  use the generic interface for determining link.
 **/
static s32 igb_check_for_link_82575(struct e1000_hw *hw)
{
	s32 ret_val;
	u16 speed, duplex;

	DEBUGFUNC("igb_check_for_link_82575");

	if (hw->phy.media_type != e1000_media_type_copper) {
		ret_val = igb_get_pcs_speed_and_duplex_82575(hw, &speed,
		                                               &duplex);
		/*
		 * Use this flag to determine if link needs to be checked or
		 * not.  If we have link clear the flag so that we do not
		 * continue to check for link.
		 */
		hw->mac.get_link_status = !hw->mac.serdes_has_link;
	} else {
		ret_val = igb_check_for_copper_link_generic(hw);
	}

	return ret_val;
}

/**
 *  igb_get_pcs_speed_and_duplex_82575 - Retrieve current speed/duplex
 *  @hw: pointer to the HW structure
 *  @speed: stores the current speed
 *  @duplex: stores the current duplex
 *
 *  Using the physical coding sub-layer (PCS), retrieve the current speed and
 *  duplex, then store the values in the pointers provided.
 **/
static s32 igb_get_pcs_speed_and_duplex_82575(struct e1000_hw *hw,
                                                u16 *speed, u16 *duplex)
{
	struct e1000_mac_info *mac = &hw->mac;
	u32 pcs;

	DEBUGFUNC("igb_get_pcs_speed_and_duplex_82575");

	/* Set up defaults for the return values of this function */
	mac->serdes_has_link = false;
	*speed = 0;
	*duplex = 0;

	/*
	 * Read the PCS Status register for link state. For non-copper mode,
	 * the status register is not accurate. The PCS status register is
	 * used instead.
	 */
	pcs = E1000_READ_REG(hw, E1000_PCS_LSTAT);

	/*
	 * The link up bit determines when link is up on autoneg. The sync ok
	 * gets set once both sides sync up and agree upon link. Stable link
	 * can be determined by checking for both link up and link sync ok
	 */
	if ((pcs & E1000_PCS_LSTS_LINK_OK) && (pcs & E1000_PCS_LSTS_SYNK_OK)) {
		mac->serdes_has_link = true;

		/* Detect and store PCS speed */
		if (pcs & E1000_PCS_LSTS_SPEED_1000) {
			*speed = SPEED_1000;
		} else if (pcs & E1000_PCS_LSTS_SPEED_100) {
			*speed = SPEED_100;
		} else {
			*speed = SPEED_10;
		}

		/* Detect and store PCS duplex */
		if (pcs & E1000_PCS_LSTS_DUPLEX_FULL) {
			*duplex = FULL_DUPLEX;
		} else {
			*duplex = HALF_DUPLEX;
		}
	}

	return E1000_SUCCESS;
}

/**
 *  igb_shutdown_serdes_link_82575 - Remove link during power down
 *  @hw: pointer to the HW structure
 *
 *  In the case of serdes shut down sfp and PCS on driver unload
 *  when management pass thru is not enabled.
 **/
void igb_shutdown_serdes_link_82575(struct e1000_hw *hw)
{
#if 0
	u32 reg;
#endif
	u16 eeprom_data = 0;

	if ((hw->phy.media_type != e1000_media_type_internal_serdes) &&
	    !igb_sgmii_active_82575(hw))
		return;

	if (hw->bus.func == E1000_FUNC_0)
		hw->nvm.ops.read(hw, NVM_INIT_CONTROL3_PORT_A, 1, &eeprom_data);
	else if (hw->bus.func == E1000_FUNC_1)
		hw->nvm.ops.read(hw, NVM_INIT_CONTROL3_PORT_B, 1, &eeprom_data);

	/*
	 * If APM is not enabled in the EEPROM and management interface is
	 * not enabled, then power down.
	 */
#if 0
	if (!(eeprom_data & E1000_NVM_APME_82575) &&
	    !igb_enable_mng_pass_thru(hw)) {
		/* Disable PCS to turn off link */
		reg = E1000_READ_REG(hw, E1000_PCS_CFG0);
		reg &= ~E1000_PCS_CFG_PCS_EN;
		E1000_WRITE_REG(hw, E1000_PCS_CFG0, reg);

		/* shutdown the laser */
		reg = E1000_READ_REG(hw, E1000_CTRL_EXT);
		reg |= E1000_CTRL_EXT_SDP3_DATA;
		E1000_WRITE_REG(hw, E1000_CTRL_EXT, reg);

		/* flush the write to verify completion */
		E1000_WRITE_FLUSH(hw);
		msec_delay(1);
	}
#endif
	return;
}

/**
 *  igb_reset_hw_82575 - Reset hardware
 *  @hw: pointer to the HW structure
 *
 *  This resets the hardware into a known state.
 **/
static s32 igb_reset_hw_82575(struct e1000_hw *hw)
{
	u32 ctrl;
	s32 ret_val;

	DEBUGFUNC("igb_reset_hw_82575");

	/*
	 * Prevent the PCI-E bus from sticking if there is no TLP connection
	 * on the last TLP read/write transaction when MAC is reset.
	 */
	ret_val = igb_disable_pcie_master_generic(hw);
	if (ret_val) {
		DEBUGOUT("PCI-E Master disable polling has failed.\n");
	}

	/* set the completion timeout for interface */
	ret_val = igb_set_pcie_completion_timeout(hw);
	if (ret_val) {
		DEBUGOUT("PCI-E Set completion timeout has failed.\n");
	}

	DEBUGOUT("Masking off all interrupts\n");
	E1000_WRITE_REG(hw, E1000_IMC, 0xffffffff);

	E1000_WRITE_REG(hw, E1000_RCTL, 0);
	E1000_WRITE_REG(hw, E1000_TCTL, E1000_TCTL_PSP);
	E1000_WRITE_FLUSH(hw);

	msec_delay(10);

	ctrl = E1000_READ_REG(hw, E1000_CTRL);

	DEBUGOUT("Issuing a global reset to MAC\n");
	E1000_WRITE_REG(hw, E1000_CTRL, ctrl | E1000_CTRL_RST);

	ret_val = igb_get_auto_rd_done_generic(hw);
	if (ret_val) {
		/*
		 * When auto config read does not complete, do not
		 * return with an error. This can happen in situations
		 * where there is no eeprom and prevents getting link.
		 */
		DEBUGOUT("Auto Read Done did not complete\n");
	}

	/* If EEPROM is not present, run manual init scripts */
	if ((E1000_READ_REG(hw, E1000_EECD) & E1000_EECD_PRES) == 0)
		igb_reset_init_script_82575(hw);

	/* Clear any pending interrupt events. */
	E1000_WRITE_REG(hw, E1000_IMC, 0xffffffff);
	E1000_READ_REG(hw, E1000_ICR);

	/* Install any alternate MAC address into RAR0 */
	ret_val = igb_check_alt_mac_addr_generic(hw);

	return ret_val;
}

/**
 *  igb_init_hw_82575 - Initialize hardware
 *  @hw: pointer to the HW structure
 *
 *  This inits the hardware readying it for operation.
 **/
static s32 igb_init_hw_82575(struct e1000_hw *hw)
{
	struct e1000_mac_info *mac = &hw->mac;
	s32 ret_val;
	u16 i, rar_count = mac->rar_entry_count;

	DEBUGFUNC("igb_init_hw_82575");

	/* Initialize identification LED */
	ret_val = mac->ops.id_led_init(hw);
	if (ret_val) {
		DEBUGOUT("Error initializing identification LED\n");
		/* This is not fatal and we should not stop init due to this */
	}

	/* Disabling VLAN filtering */
	DEBUGOUT("Initializing the IEEE VLAN\n");
	mac->ops.clear_vfta(hw);

	/* Setup the receive address */
	igb_init_rx_addrs_generic(hw, rar_count);

	/* Zero out the Multicast HASH table */
	DEBUGOUT("Zeroing the MTA\n");
	for (i = 0; i < mac->mta_reg_count; i++)
		E1000_WRITE_REG_ARRAY(hw, E1000_MTA, i, 0);

	/* Zero out the Unicast HASH table */
	DEBUGOUT("Zeroing the UTA\n");
	for (i = 0; i < mac->uta_reg_count; i++)
		E1000_WRITE_REG_ARRAY(hw, E1000_UTA, i, 0);

	/* Setup link and flow control */
	ret_val = mac->ops.setup_link(hw);

	/*
	 * Clear all of the statistics registers (clear on read).  It is
	 * important that we do this after we have tried to establish link
	 * because the symbol error count will increment wildly if there
	 * is no link.
	 */
	igb_clear_hw_cntrs_82575(hw);

	return ret_val;
}

/**
 *  igb_setup_copper_link_82575 - Configure copper link settings
 *  @hw: pointer to the HW structure
 *
 *  Configures the link for auto-neg or forced speed and duplex.  Then we check
 *  for link, once link is established calls to configure collision distance
 *  and flow control are called.
 **/
static s32 igb_setup_copper_link_82575(struct e1000_hw *hw)
{
	u32 ctrl;
	s32  ret_val;

	DEBUGFUNC("igb_setup_copper_link_82575");

	ctrl = E1000_READ_REG(hw, E1000_CTRL);
	ctrl |= E1000_CTRL_SLU;
	ctrl &= ~(E1000_CTRL_FRCSPD | E1000_CTRL_FRCDPX);
	E1000_WRITE_REG(hw, E1000_CTRL, ctrl);

	ret_val = igb_setup_serdes_link_82575(hw);
	if (ret_val)
		goto out;

	if (igb_sgmii_active_82575(hw) && !hw->phy.reset_disable) {
		ret_val = hw->phy.ops.reset(hw);
		if (ret_val) {
			DEBUGOUT("Error resetting the PHY.\n");
			goto out;
		}
	}
	switch (hw->phy.type) {
	case e1000_phy_m88:
		ret_val = igb_copper_link_setup_m88(hw);
		break;
	case e1000_phy_igp_3:
		ret_val = igb_copper_link_setup_igp(hw);
		break;
	default:
		ret_val = -E1000_ERR_PHY;
		break;
	}

	if (ret_val)
		goto out;

	ret_val = igb_setup_copper_link_generic(hw);
out:
	return ret_val;
}

/**
 *  igb_setup_serdes_link_82575 - Setup link for serdes
 *  @hw: pointer to the HW structure
 *
 *  Configure the physical coding sub-layer (PCS) link.  The PCS link is
 *  used on copper connections where the serialized gigabit media independent
 *  interface (sgmii), or serdes fiber is being used.  Configures the link
 *  for auto-negotiation or forces speed/duplex.
 **/
static s32 igb_setup_serdes_link_82575(struct e1000_hw *hw)
{
	u32 ctrl_reg, reg;

	DEBUGFUNC("igb_setup_serdes_link_82575");

	if ((hw->phy.media_type != e1000_media_type_internal_serdes) &&
	    !igb_sgmii_active_82575(hw))
		return E1000_SUCCESS;

	/*
	 * On the 82575, SerDes loopback mode persists until it is
	 * explicitly turned off or a power cycle is performed.  A read to
	 * the register does not indicate its status.  Therefore, we ensure
	 * loopback mode is disabled during initialization.
	 */
	E1000_WRITE_REG(hw, E1000_SCTL, E1000_SCTL_DISABLE_SERDES_LOOPBACK);

	/* power on the sfp cage if present */
	reg = E1000_READ_REG(hw, E1000_CTRL_EXT);
	reg &= ~E1000_CTRL_EXT_SDP3_DATA;
	E1000_WRITE_REG(hw, E1000_CTRL_EXT, reg);

	ctrl_reg = E1000_READ_REG(hw, E1000_CTRL);
	ctrl_reg |= E1000_CTRL_SLU;

	if (hw->mac.type == e1000_82575 || hw->mac.type == e1000_82576) {
		/* set both sw defined pins */
		ctrl_reg |= E1000_CTRL_SWDPIN0 | E1000_CTRL_SWDPIN1;

		/* Set switch control to serdes energy detect */
		reg = E1000_READ_REG(hw, E1000_CONNSW);
		reg |= E1000_CONNSW_ENRGSRC;
		E1000_WRITE_REG(hw, E1000_CONNSW, reg);
	}

	reg = E1000_READ_REG(hw, E1000_PCS_LCTL);

	if (igb_sgmii_active_82575(hw)) {
		/* allow time for SFP cage to power up phy */
		msec_delay(300);

		/* AN time out should be disabled for SGMII mode */
		reg &= ~(E1000_PCS_LCTL_AN_TIMEOUT);
	} else {
		ctrl_reg |= E1000_CTRL_SPD_1000 | E1000_CTRL_FRCSPD |
		            E1000_CTRL_FD | E1000_CTRL_FRCDPX;
	}

	E1000_WRITE_REG(hw, E1000_CTRL, ctrl_reg);

	/*
	 * New SerDes mode allows for forcing speed or autonegotiating speed
	 * at 1gb. Autoneg should be default set by most drivers. This is the
	 * mode that will be compatible with older link partners and switches.
	 * However, both are supported by the hardware and some drivers/tools.
	 */

	reg &= ~(E1000_PCS_LCTL_AN_ENABLE | E1000_PCS_LCTL_FLV_LINK_UP |
	         E1000_PCS_LCTL_FSD | E1000_PCS_LCTL_FORCE_LINK);

	/*
	 * We force flow control to prevent the CTRL register values from being
	 * overwritten by the autonegotiated flow control values
	 */
	reg |= E1000_PCS_LCTL_FORCE_FCTRL;

	/*
	 * we always set sgmii to autoneg since it is the phy that will be
	 * forcing the link and the serdes is just a go-between
	 */
	if (hw->mac.autoneg || igb_sgmii_active_82575(hw)) {
		/* Set PCS register for autoneg */
		reg |= E1000_PCS_LCTL_FSV_1000 |  /* Force 1000 */
		       E1000_PCS_LCTL_FDV_FULL |  /* SerDes Full dplx */
		       E1000_PCS_LCTL_AN_ENABLE | /* Enable Autoneg */
		       E1000_PCS_LCTL_AN_RESTART; /* Restart autoneg */
		DEBUGOUT1("Configuring Autoneg:PCS_LCTL=0x%08X\n", reg);
	} else {
		/* Check for duplex first */
		if (hw->mac.forced_speed_duplex & E1000_ALL_FULL_DUPLEX)
			reg |= E1000_PCS_LCTL_FDV_FULL;

		/* No need to check for 1000/full since the spec states that
		 * it requires autoneg to be enabled */
		/* Now set speed */
		if (hw->mac.forced_speed_duplex & E1000_ALL_100_SPEED)
			reg |= E1000_PCS_LCTL_FSV_100;

		/* Force speed and force link */
		reg |= E1000_PCS_LCTL_FSD |
		       E1000_PCS_LCTL_FORCE_LINK |
		       E1000_PCS_LCTL_FLV_LINK_UP;

		DEBUGOUT1("Configuring Forced Link:PCS_LCTL=0x%08X\n", reg);
	}

	E1000_WRITE_REG(hw, E1000_PCS_LCTL, reg);

	if (!igb_sgmii_active_82575(hw))
		igb_force_mac_fc_generic(hw);

	return E1000_SUCCESS;
}

/**
 *  igb_valid_led_default_82575 - Verify a valid default LED config
 *  @hw: pointer to the HW structure
 *  @data: pointer to the NVM (EEPROM)
 *
 *  Read the EEPROM for the current default LED configuration.  If the
 *  LED configuration is not valid, set to a valid LED configuration.
 **/
static s32 igb_valid_led_default_82575(struct e1000_hw *hw, u16 *data)
{
	s32 ret_val;

	DEBUGFUNC("igb_valid_led_default_82575");

	ret_val = hw->nvm.ops.read(hw, NVM_ID_LED_SETTINGS, 1, data);
	if (ret_val) {
		DEBUGOUT("NVM Read Error\n");
		goto out;
	}

	if (*data == ID_LED_RESERVED_0000 || *data == ID_LED_RESERVED_FFFF) {
		switch(hw->phy.media_type) {
		case e1000_media_type_internal_serdes:
			*data = ID_LED_DEFAULT_82575_SERDES;
			break;
		case e1000_media_type_copper:
		default:
			*data = ID_LED_DEFAULT;
			break;
		}
	}
out:
	return ret_val;
}

/**
 *  igb_sgmii_active_82575 - Return sgmii state
 *  @hw: pointer to the HW structure
 *
 *  82575 silicon has a serialized gigabit media independent interface (sgmii)
 *  which can be enabled for use in the embedded applications.  Simply
 *  return the current state of the sgmii interface.
 **/
static bool igb_sgmii_active_82575(struct e1000_hw *hw)
{
	struct e1000_dev_spec_82575 *dev_spec = &hw->dev_spec._82575;
	return dev_spec->sgmii_active;
}

/**
 *  igb_reset_init_script_82575 - Inits HW defaults after reset
 *  @hw: pointer to the HW structure
 *
 *  Inits recommended HW defaults after a reset when there is no EEPROM
 *  detected. This is only for the 82575.
 **/
static s32 igb_reset_init_script_82575(struct e1000_hw* hw)
{
	DEBUGFUNC("igb_reset_init_script_82575");

	if (hw->mac.type == e1000_82575) {
		DEBUGOUT("Running reset init script for 82575\n");
		/* SerDes configuration via SERDESCTRL */
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCTL, 0x00, 0x0C);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCTL, 0x01, 0x78);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCTL, 0x1B, 0x23);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCTL, 0x23, 0x15);

		/* CCM configuration via CCMCTL register */
		igb_write_8bit_ctrl_reg_generic(hw, E1000_CCMCTL, 0x14, 0x00);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_CCMCTL, 0x10, 0x00);

		/* PCIe lanes configuration */
		igb_write_8bit_ctrl_reg_generic(hw, E1000_GIOCTL, 0x00, 0xEC);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_GIOCTL, 0x61, 0xDF);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_GIOCTL, 0x34, 0x05);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_GIOCTL, 0x2F, 0x81);

		/* PCIe PLL Configuration */
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCCTL, 0x02, 0x47);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCCTL, 0x14, 0x00);
		igb_write_8bit_ctrl_reg_generic(hw, E1000_SCCTL, 0x10, 0x00);
	}

	return E1000_SUCCESS;
}

/**
 *  igb_read_mac_addr_82575 - Read device MAC address
 *  @hw: pointer to the HW structure
 **/
static s32 igb_read_mac_addr_82575(struct e1000_hw *hw)
{
	s32 ret_val = E1000_SUCCESS;

	DEBUGFUNC("igb_read_mac_addr_82575");

	/*
	 * If there's an alternate MAC address place it in RAR0
	 * so that it will override the Si installed default perm
	 * address.
	 */
	ret_val = igb_check_alt_mac_addr_generic(hw);
	if (ret_val)
		goto out;

	ret_val = igb_read_mac_addr_generic(hw);

out:
	return ret_val;
}

/**
 * igb_power_down_phy_copper_82575 - Remove link during PHY power down
 * @hw: pointer to the HW structure
 *
 * In the case of a PHY power down to save power, or to turn off link during a
 * driver unload, or wake on lan is not enabled, remove the link.
 **/
static void igb_power_down_phy_copper_82575(struct e1000_hw *hw)
{
	struct e1000_phy_info *phy = &hw->phy;
	struct e1000_mac_info *mac = &hw->mac;

	if (!(phy->ops.check_reset_block))
		return;

	/* If the management interface is not enabled, then power down */
	if (!(mac->ops.check_mng_mode(hw) || phy->ops.check_reset_block(hw)))
		igb_power_down_phy_copper(hw);

	return;
}

/**
 *  igb_clear_hw_cntrs_82575 - Clear device specific hardware counters
 *  @hw: pointer to the HW structure
 *
 *  Clears the hardware counters by reading the counter registers.
 **/
static void igb_clear_hw_cntrs_82575(struct e1000_hw *hw)
{
	DEBUGFUNC("igb_clear_hw_cntrs_82575");

	igb_clear_hw_cntrs_base_generic(hw);

	E1000_READ_REG(hw, E1000_PRC64);
	E1000_READ_REG(hw, E1000_PRC127);
	E1000_READ_REG(hw, E1000_PRC255);
	E1000_READ_REG(hw, E1000_PRC511);
	E1000_READ_REG(hw, E1000_PRC1023);
	E1000_READ_REG(hw, E1000_PRC1522);
	E1000_READ_REG(hw, E1000_PTC64);
	E1000_READ_REG(hw, E1000_PTC127);
	E1000_READ_REG(hw, E1000_PTC255);
	E1000_READ_REG(hw, E1000_PTC511);
	E1000_READ_REG(hw, E1000_PTC1023);
	E1000_READ_REG(hw, E1000_PTC1522);

	E1000_READ_REG(hw, E1000_ALGNERRC);
	E1000_READ_REG(hw, E1000_RXERRC);
	E1000_READ_REG(hw, E1000_TNCRS);
	E1000_READ_REG(hw, E1000_CEXTERR);
	E1000_READ_REG(hw, E1000_TSCTC);
	E1000_READ_REG(hw, E1000_TSCTFC);

	E1000_READ_REG(hw, E1000_MGTPRC);
	E1000_READ_REG(hw, E1000_MGTPDC);
	E1000_READ_REG(hw, E1000_MGTPTC);

	E1000_READ_REG(hw, E1000_IAC);
	E1000_READ_REG(hw, E1000_ICRXOC);

	E1000_READ_REG(hw, E1000_ICRXPTC);
	E1000_READ_REG(hw, E1000_ICRXATC);
	E1000_READ_REG(hw, E1000_ICTXPTC);
	E1000_READ_REG(hw, E1000_ICTXATC);
	E1000_READ_REG(hw, E1000_ICTXQEC);
	E1000_READ_REG(hw, E1000_ICTXQMTC);
	E1000_READ_REG(hw, E1000_ICRXDMTC);

	E1000_READ_REG(hw, E1000_CBTMPC);
	E1000_READ_REG(hw, E1000_HTDPMC);
	E1000_READ_REG(hw, E1000_CBRMPC);
	E1000_READ_REG(hw, E1000_RPTHC);
	E1000_READ_REG(hw, E1000_HGPTC);
	E1000_READ_REG(hw, E1000_HTCBDPC);
	E1000_READ_REG(hw, E1000_HGORCL);
	E1000_READ_REG(hw, E1000_HGORCH);
	E1000_READ_REG(hw, E1000_HGOTCL);
	E1000_READ_REG(hw, E1000_HGOTCH);
	E1000_READ_REG(hw, E1000_LENERRS);

	/* This register should not be read in copper configurations */
	if ((hw->phy.media_type == e1000_media_type_internal_serdes) ||
	    igb_sgmii_active_82575(hw))
		E1000_READ_REG(hw, E1000_SCVPC);
}

/**
 *  igb_rx_fifo_flush_82575 - Clean rx fifo after RX enable
 *  @hw: pointer to the HW structure
 *
 *  After rx enable if managability is enabled then there is likely some
 *  bad data at the start of the fifo and possibly in the DMA fifo.  This
 *  function clears the fifos and flushes any packets that came in as rx was
 *  being enabled.
 **/
void igb_rx_fifo_flush_82575(struct e1000_hw *hw)
{
	u32 rctl, rlpml, rxdctl[4], rfctl, temp_rctl, rx_enabled;
	int i, ms_wait;

	DEBUGFUNC("igb_rx_fifo_workaround_82575");
	if (hw->mac.type != e1000_82575 ||
	    !(E1000_READ_REG(hw, E1000_MANC) & E1000_MANC_RCV_TCO_EN))
		return;

	/* Disable all RX queues */
	for (i = 0; i < 4; i++) {
		rxdctl[i] = E1000_READ_REG(hw, E1000_RXDCTL(i));
		E1000_WRITE_REG(hw, E1000_RXDCTL(i),
		                rxdctl[i] & ~E1000_RXDCTL_QUEUE_ENABLE);
	}
	/* Poll all queues to verify they have shut down */
	for (ms_wait = 0; ms_wait < 10; ms_wait++) {
		msec_delay(1);
		rx_enabled = 0;
		for (i = 0; i < 4; i++)
			rx_enabled |= E1000_READ_REG(hw, E1000_RXDCTL(i));
		if (!(rx_enabled & E1000_RXDCTL_QUEUE_ENABLE))
			break;
	}

	if (ms_wait == 10) {
		DEBUGOUT("Queue disable timed out after 10ms\n");
        }
	/* Clear RLPML, RCTL.SBP, RFCTL.LEF, and set RCTL.LPE so that all
	 * incoming packets are rejected.  Set enable and wait 2ms so that
	 * any packet that was coming in as RCTL.EN was set is flushed
	 */
	rfctl = E1000_READ_REG(hw, E1000_RFCTL);
	E1000_WRITE_REG(hw, E1000_RFCTL, rfctl & ~E1000_RFCTL_LEF);

	rlpml = E1000_READ_REG(hw, E1000_RLPML);
	E1000_WRITE_REG(hw, E1000_RLPML, 0);

	rctl = E1000_READ_REG(hw, E1000_RCTL);
	temp_rctl = rctl & ~(E1000_RCTL_EN | E1000_RCTL_SBP);
	temp_rctl |= E1000_RCTL_LPE;

	E1000_WRITE_REG(hw, E1000_RCTL, temp_rctl);
	E1000_WRITE_REG(hw, E1000_RCTL, temp_rctl | E1000_RCTL_EN);
	E1000_WRITE_FLUSH(hw);
	msec_delay(2);

	/* Enable RX queues that were previously enabled and restore our
	 * previous state
	 */
	for (i = 0; i < 4; i++)
		E1000_WRITE_REG(hw, E1000_RXDCTL(i), rxdctl[i]);
	E1000_WRITE_REG(hw, E1000_RCTL, rctl);
	E1000_WRITE_FLUSH(hw);

	E1000_WRITE_REG(hw, E1000_RLPML, rlpml);
	E1000_WRITE_REG(hw, E1000_RFCTL, rfctl);

	/* Flush receive errors generated by workaround */
	E1000_READ_REG(hw, E1000_ROC);
	E1000_READ_REG(hw, E1000_RNBC);
	E1000_READ_REG(hw, E1000_MPC);
}

/**
 *  igb_set_pcie_completion_timeout - set pci-e completion timeout
 *  @hw: pointer to the HW structure
 *
 *  The defaults for 82575 and 82576 should be in the range of 50us to 50ms,
 *  however the hardware default for these parts is 500us to 1ms which is less
 *  than the 10ms recommended by the pci-e spec.  To address this we need to
 *  increase the value to either 10ms to 200ms for capability version 1 config,
 *  or 16ms to 55ms for version 2.
 **/
static s32 igb_set_pcie_completion_timeout(struct e1000_hw *hw)
{
	u32 gcr = E1000_READ_REG(hw, E1000_GCR);
	s32 ret_val = E1000_SUCCESS;
	u16 pcie_devctl2;

	/* only take action if timeout value is defaulted to 0 */
	if (gcr & E1000_GCR_CMPL_TMOUT_MASK)
		goto out;

	/*
	 * if capababilities version is type 1 we can write the
	 * timeout of 10ms to 200ms through the GCR register
	 */
	if (!(gcr & E1000_GCR_CAP_VER2)) {
		gcr |= E1000_GCR_CMPL_TMOUT_10ms;
		goto out;
	}

	/*
	 * for version 2 capabilities we need to write the config space
	 * directly in order to set the completion timeout value for
	 * 16ms to 55ms
	 */
	ret_val = igb_read_pcie_cap_reg(hw, PCIE_DEVICE_CONTROL2,
	                                  &pcie_devctl2);
	if (ret_val)
		goto out;

	pcie_devctl2 |= PCIE_DEVICE_CONTROL2_16ms;

	ret_val = igb_write_pcie_cap_reg(hw, PCIE_DEVICE_CONTROL2,
	                                   &pcie_devctl2);
out:
	/* disable completion timeout resend */
	gcr &= ~E1000_GCR_CMPL_TMOUT_RESEND;

	E1000_WRITE_REG(hw, E1000_GCR, gcr);
	return ret_val;
}

/**
 *  igb_vmdq_set_loopback_pf - enable or disable vmdq loopback
 *  @hw: pointer to the hardware struct
 *  @enable: state to enter, either enabled or disabled
 *
 *  enables/disables L2 switch loopback functionality.
 **/
void igb_vmdq_set_loopback_pf(struct e1000_hw *hw, bool enable)
{
	u32 dtxswc = E1000_READ_REG(hw, E1000_DTXSWC);

	if (enable)
		dtxswc |= E1000_DTXSWC_VMDQ_LOOPBACK_EN;
	else
		dtxswc &= ~E1000_DTXSWC_VMDQ_LOOPBACK_EN;

	E1000_WRITE_REG(hw, E1000_DTXSWC, dtxswc);
}

/**
 *  igb_vmdq_set_replication_pf - enable or disable vmdq replication
 *  @hw: pointer to the hardware struct
 *  @enable: state to enter, either enabled or disabled
 *
 *  enables/disables replication of packets across multiple pools.
 **/
void igb_vmdq_set_replication_pf(struct e1000_hw *hw, bool enable)
{
	u32 vt_ctl = E1000_READ_REG(hw, E1000_VT_CTL);

	if (enable)
		vt_ctl |= E1000_VT_CTL_VM_REPL_EN;
	else
		vt_ctl &= ~E1000_VT_CTL_VM_REPL_EN;

	E1000_WRITE_REG(hw, E1000_VT_CTL, vt_ctl);
}

static struct pci_device_id igb_82575_nics[] = {
        PCI_ROM(0x8086, 0x10C9, "E1000_DEV_ID_82576", "E1000_DEV_ID_82576", 0),
        PCI_ROM(0x8086, 0x150A, "E1000_DEV_ID_82576_NS", "E1000_DEV_ID_82576_NS", 0),
        PCI_ROM(0x8086, 0x1518, "E1000_DEV_ID_82576_NS_SERDES", "E1000_DEV_ID_82576_NS_SERDES", 0),
        PCI_ROM(0x8086, 0x10E6, "E1000_DEV_ID_82576_FIBER", "E1000_DEV_ID_82576_FIBER", 0),
        PCI_ROM(0x8086, 0x10E7, "E1000_DEV_ID_82576_SERDES", "E1000_DEV_ID_82576_SERDES", 0),
        PCI_ROM(0x8086, 0x150D, "E1000_DEV_ID_82576_SERDES_QUAD", "E1000_DEV_ID_82576_SERDES_QUAD", 0),
        PCI_ROM(0x8086, 0x10E8, "E1000_DEV_ID_82576_QUAD_COPPER", "E1000_DEV_ID_82576_QUAD_COPPER", 0),
        PCI_ROM(0x8086, 0x10A7, "E1000_DEV_ID_82575EB_COPPER", "E1000_DEV_ID_82575EB_COPPER", 0),
        PCI_ROM(0x8086, 0x10A9, "E1000_DEV_ID_82575EB_FIBER_SERDES", "E1000_DEV_ID_82575EB_FIBER_SERDES", 0),
        PCI_ROM(0x8086, 0x10D6, "E1000_DEV_ID_82575GB_QUAD_COPPER", "E1000_DEV_ID_82575GB_QUAD_COPPER", 0),
};

struct pci_driver igb_82575_driver __pci_driver = {
        .ids = igb_82575_nics,
        .id_count = (sizeof (igb_82575_nics) / sizeof (igb_82575_nics[0])),
        .probe = igb_probe,
        .remove = igb_remove,
};
